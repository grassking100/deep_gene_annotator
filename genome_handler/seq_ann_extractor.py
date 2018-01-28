from . import numpy as np
from . import deepdish
#Purpose:Get selected sequences annotation by sequences info
#Input:Sequences info and annotated genome sequences
#Output:Selected sequences annotation
class SeqAnnExtractor:
    def save(self,ann_seqs,file_name,to_dictionary=False):
        if  not to_dictionary:
            file_to_save=ann_seqs
        else:
            ann_dict={}
            for ann_seq in ann_seqs:
                temp={}
                redirected_ann_seq=self.redirection_seq(ann_seq)                
                for ann_type in self.__ANNOTATION_TYPES:
                    temp[ann_type]=redirected_ann_seq[ann_type]
                ann_dict[str(ann_seq['id'])]=np.transpose(temp)
            file_to_save=ann_dict
        deepdish.io.save(file_name,file_to_save)
    def redirection_seq(self,ann_seq):
        temp={}
        temp['id']=ann_seq['id']
        temp['strand']=ann_seq['strand']
        for annotation_type in self.__ANNOTATION_TYPES:
            if temp['strand']=='-':
                temp[annotation_type]=np.fliplr([ann_seq[annotation_type]])[0]
            else:
                temp[annotation_type]=ann_seq[annotation_type]
        return temp
    def redirection_seqs(self,ann_seqs):
        temp=[]
        for seq in ann_seqs:
            temp.append(self.redirection_seq(seq))
        return temp
    def __init__(self,seqInfoExtractor,genomeAnnotator,is_normalized):
        self.__ANNOTATION_TYPES=['utr_5','utr_3','intron','cds','intergenic_region']
        self.__sequences_info=seqInfoExtractor.sequences_info
        self.__annotated_genome=genomeAnnotator.get_genome(is_normalized)
        self.__selected_sequences_annotation=[]
        self.__extract_sequences()
    @property
    def selected_sequences_annotation(self):
        return self.__selected_sequences_annotation
    def __extract_sequence(self,sequence_info):
        start=sequence_info['start']
        end=sequence_info['end']
        chrom=sequence_info['chrom']
        strand=sequence_info['strand']
        mini_ann_seq={}
        mini_ann_seq['strand']=sequence_info['strand']
        mini_ann_seq['id']=sequence_info['id']
        for annotation_type in self.__ANNOTATION_TYPES:
            seq=self.__annotated_genome[chrom][strand][annotation_type][start:end+1]
            mini_ann_seq[annotation_type]=seq
        return mini_ann_seq
    def __extract_sequences(self):
        for seq_info in self.__sequences_info:
            seq_ann=self.__extract_sequence(seq_info)
            self.__selected_sequences_annotation.append(seq_ann)
