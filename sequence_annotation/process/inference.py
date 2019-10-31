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

def ann_seq2one_hot_seq(ann_seq,length=None):
    C,L = ann_seq.shape
    index = ann_seq.argmax(0)
    if length is not None:
        index = index[:length]
    return index2one_hot(index,C)

class AnnSeq2InfoConverter:
    def __init__(self,ann_types,simplify_map,dist,
                 donor_site_pattern=None,accept_site_pattern=None):
        self.extractor = GeneInfoExtractor()
        self.donor_site_pattern = donor_site_pattern
        self.accept_site_pattern = accept_site_pattern
        self.ann_types = ann_types
        self.simplify_map = simplify_map
        self.dist = dist

    def convert(self,chrom_ids,seqs,ann_seqs,lengths,fix_boundary=True):
        seq_infos = SeqInfoContainer()
        for chrom_id,seq,ann_seq, length in zip(chrom_ids,seqs,ann_seqs,lengths):
            ann_seq = ann_seq2one_hot_seq(ann_seq,length)
            ann_seq = vecs2seq(ann_seq,chrom_id,'plus',self.ann_types)
            infos = self.extractor.extract_per_seq(ann_seq,self.simplify_map)
            if fix_boundary and len(infos) > 0:
                gff = infos.to_gff()
                fixed_ann_seq = fix_ann_seq(chrom_id,length,seq,gff,self.dist,
                                            self.donor_site_pattern,
                                            self.accept_site_pattern)
                infos = self.extractor.extract_per_seq(fixed_ann_seq,self.simplify_map)
            seq_infos.add(infos)
        return seq_infos