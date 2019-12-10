import numpy as np
import torch
from ..genome_handler.seq_container import AnnSeqContainer,SeqInfoContainer
from ..genome_handler.region_extractor import GeneInfoExtractor
from ..genome_handler.ann_seq_processor import vecs2seq
from .boundary_process import fix_ann_seq

def basic_inference(first_n_channel,before=True):
    def _inference(ann,mask=None):
        if first_n_channel > ann.shape[1]:
            raise Exception("Wrong channel size, got {} and {}".format(first_n_channel,ann.shape[1]))
        if before:
            ann = ann[:,:first_n_channel,:]
        else:
            ann = ann[:,first_n_channel:,:]
        if mask is not None:
            ann = ann.transpose(0,1)
            L = ann.shape[2]
            ann = ann*(mask[:,:L].to(ann.dtype))
            ann = ann.transpose(0,1)
        return ann
    return _inference

def seq_ann_inference(ann,mask):
    """
        Data shape is N,C,L (where C>=2)
        Input channel order: Transcription potential, Intron potential,*
        Output channel order: Exon, Intron , Other
    """
    if ann.shape[1] != 2:
        raise Exception("Channel size should be equal to two, but got {}".format(ann.shape[1]))
    transcript_potential = ann[:,0,:].unsqueeze(1)
    intron_potential = ann[:,1,:].unsqueeze(1)
    other = 1-transcript_potential
    if mask is not None:
        mask = mask[:,:ann.shape[2]].unsqueeze(1)
        other = other * mask.float()
    transcript_mask = (transcript_potential>=0.5).float()
    intron = transcript_mask * intron_potential
    exon = transcript_mask * (1-intron_potential)
    result = torch.cat([exon,intron,other],dim=1)
    return result

def seq_ann_reverse_inference(ann,mask):
    """
        Data shape is N,C,L
        Input channel order: Exon, Intron , Other,*
        Output channel order: Transcription potential, Intron potential
    """
    if ann.shape[1] != 3:
        raise Exception("Channel size should be eqaul to three")
    intron_potential = ann[:,1,:].unsqueeze(1)
    other_potential = ann[:,2,:].unsqueeze(1)
    transcript_potential = 1 - other_potential
    result = torch.cat([transcript_potential,intron_potential],dim=1)
    return result

def index2one_hot(index,channel_size):
    if (np.array(index)<0).any() or (np.array(index)>=channel_size).any():
        raise Exception("Invalid number")
    L = len(index)
    loc = list(range(L))
    onehot = np.zeros((channel_size,L))
    onehot[index,loc]=1
    return onehot

def ann_vec2one_hot_vec(ann_vec,length=None):
    C,L = ann_vec.shape
    index = ann_vec.argmax(0)
    if length is not None:
        index = index[:length]
    return index2one_hot(index,C)

class AnnVec2InfoConverter:
    def __init__(self,ann_types,gene_map,dist=None,
                 donor_site_pattern=None,accept_site_pattern=None):
        self.extractor = GeneInfoExtractor()
        self.donor_site_pattern = donor_site_pattern
        self.accept_site_pattern = accept_site_pattern
        self.ann_types = ann_types
        self.gene_map = gene_map
        self.dist = dist or 16
        
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['donor_site_pattern'] = self.donor_site_pattern
        config['accept_site_pattern'] = self.accept_site_pattern
        config['ann_types'] = self.ann_types
        config['gene_map'] = self.gene_map
        config['dist'] = self.dist
        config['alt'] = self.extractor.alt
        config['alt_num'] = self.extractor.alt_num
        return config
        
    def _fixed_info(self,chrom_id,length,dna_seq,gff):
        """Convert GFF to SeqInfoContainer about fixed region data"""
        fixed_ann_seq = fix_ann_seq(chrom_id,length,dna_seq,gff,self.dist,
                                    self.donor_site_pattern,self.accept_site_pattern)
        info = self.extractor.extract_per_seq(fixed_ann_seq,self.gene_map)
        return info
        
    def _info_dict2fixed_info(self,chrom_ids,lengths,dna_seqs,seq_info_dict):
        """Convert dictionay of SeqInformation to SeqInfoContainer about fixed region data"""
        returned = SeqInfoContainer()
        for chrom_id,dna_seq, length in zip(chrom_ids,dna_seqs,lengths):
            info = seq_info_dict[chrom_id]
            if len(info) > 0:
                gff = info.to_gff()
                info = self._fixed_info(chrom_id,length,dna_seq,gff)
                returned.add(info)
        return returned

    def _vecs2info_dict(self,chrom_ids,lengths,ann_vecs):
        """Convert annotation vectors to dictionay of SeqInformation of region data"""
        seq_info_dict = {}
        for chrom_id,ann_vec, length in zip(chrom_ids,ann_vecs,lengths):
            one_hot_vec = ann_vec2one_hot_vec(ann_vec,length)
            ann_seq = vecs2seq(one_hot_vec,chrom_id,'plus',self.ann_types)
            infos = self.extractor.extract_per_seq(ann_seq,self.gene_map)
            seq_info_dict[chrom_id] = infos
        return seq_info_dict
    
    def vecs2info(self,chrom_ids,lengths,ann_vecs):
        """Convert annotation vectors to GFF about region data"""
        returned = SeqInfoContainer()
        info_dict = self._vecs2info_dict(chrom_ids,lengths,ann_vecs)
        for seq in info_dict.values():
            returned.add(seq)
        return returned.to_gff()
    
    def vecs2fixed_info(self,chrom_ids,lengths,dna_seqs,ann_vecs):
        """Convert annotation vectors to GFF about fixed region data"""
        info_dict = self._vecs2info_dict(chrom_ids,lengths,ann_vecs)
        fixed_info = self._info_dict2fixed_info(chrom_ids,lengths,dna_seqs,info_dict)
        return info_dict.to_gff()
    
    def fix_boundary(self,gff,dna_dict):
        gff_group = gff.groupby('chr')
        chroms = set(gff['chr'])
        returned = SeqInfoContainer()
        for chrom_id in chroms:
            dna_seq = dna_dict[chrom_id]
            length = len(dna_seq)
            part_gff = gff_group.get_group(chrom_id)
            info = self._fixed_info(chrom_id,length,dna_seq,part_gff)
            returned.add(info)
        return returned.to_gff()
