import torch
import pandas as pd
import numpy as np
from ..genome_handler.seq_container import AnnSeqContainer,SeqInfoContainer,EmptyContainerException
from ..genome_handler.region_extractor import GeneInfoExtractor
from ..genome_handler.ann_seq_processor import vecs2seq
from .boundary_process import fix_ann_seq
from .post_processor import GeneAnnProcessor

def basic_inference(first_n_channel=None,before=True):
    first_n_channel = first_n_channel or 3
    def _inference(ann,mask=None):
        if first_n_channel > ann.shape[1]:
            raise Exception("Wrong channel size, got {} and {}".format(first_n_channel,ann.shape[1]))
        if before:
            ann = ann[:,:first_n_channel,:]
        else:
            ann = ann[:,first_n_channel:,:]
        if mask is not None:
            L = ann.shape[2]
            ann = ann.transpose(0,1)
            ann = ann*(mask[:,:L].to(ann.dtype))
            ann = ann.transpose(0,1)
        return ann
    return _inference

def seq_ann_inference(ann,mask,transcript_threshold=None,intron_threshold=None):
    """
        Data shape is N,C,L (where C>=2)
        Input channel order: Transcription potential, Intron potential
        Output channel order: Exon, Intron , Other
    """
    N,C,L = ann.shape
    if C != 2:
        raise Exception("Channel size should be equal to two, but got {}".format(C))
        
    if mask is not None:
        mask = mask[:,:L].unsqueeze(1).float()
        
    transcript_threshold = transcript_threshold or 0.5
    intron_threshold = intron_threshold or 0.5
    transcript_potential = ann[:,0,:].unsqueeze(1)
    intron_potential = ann[:,1,:].unsqueeze(1)
    
    transcript_mask = (transcript_potential>=transcript_threshold).float()
    intron_mask = (intron_potential>=intron_threshold).float()
    exon = transcript_mask * (1-intron_mask)
    intron = transcript_mask * intron_mask
    other = 1-transcript_potential
    if mask is not None:
        exon = exon * mask
        intron = intron * mask
        other = other * mask
    result = torch.cat([exon,intron,other],dim=1)
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
    def __init__(self,channel_order,gene_ann_processor):
        self.channel_order = channel_order
        self.gene_ann_processor = gene_ann_processor

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['channel_order'] = self.channel_order
        config['alt_num'] = self.gene_ann_processor.get_config()
        return config

    def _info_dict2fixed_gff(self,chrom_ids,lengths,dna_seqs,seq_info_dict):
        """Convert dictionay of SeqInformation to SeqInfoContainer about fixed region data"""
        returned = []
        for chrom_id,dna_seq, length in zip(chrom_ids,dna_seqs,lengths):
            info = seq_info_dict[chrom_id]
            if len(info) > 0:
                gff = info.to_gff()
                try:
                    info = self.gene_ann_processor.process(chrom_id,length,dna_seq,gff)
                    returned.append(info)
                except EmptyContainerException:
                    pass
        returned = pd.concat(returned)
        return returned

    def _vecs2info_dict(self,chrom_ids,lengths,ann_vecs):
        """Convert annotation vectors to dictionay of SeqInformation of region data"""
        seq_info_dict = {}
        for chrom_id,ann_vec, length in zip(chrom_ids,ann_vecs,lengths):
            one_hot_vec = ann_vec2one_hot_vec(ann_vec,length)
            ann_seq = vecs2seq(one_hot_vec,chrom_id,'plus',self.channel_order)
            info = self.gene_ann_processor.gene_info_extractor.extract_per_seq(ann_seq)
            seq_info_dict[chrom_id] = info
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
        fixed_gff = self._info_dict2fixed_gff(chrom_ids,lengths,dna_seqs,info_dict)
        return fixed_gff
    
    def fix_boundary(self,gff,dna_dict):
        chroms = set(gff['chr'])
        returned = []
        for chrom_id in chroms:
            dna_seq = dna_dict[chrom_id]
            length = len(dna_seq)
            info = self.gene_ann_processor.process(chrom_id,length,dna_seq,gff)
            returned.append(info)
        returned = pd.concat(returned)
        return returned
    
def build_converter(channel_order,simply_map,**kwargs):
    gene_info_extractor = GeneInfoExtractor(simply_map)
    gene_ann_processor = GeneAnnProcessor(gene_info_extractor,**kwargs)
    converter = AnnVec2InfoConverter(channel_order,gene_ann_processor)
    return converter
