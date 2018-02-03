from . import SeqInfoContainer
from . import SeqInformation
import numpy as np
class RegionExtractor:
    """#Get annotated region information"""
    def __init__(self,one_hot_ann_genome):
        if one_hot_ann_genome.note is not "one-hot":
            raise Exception("Please input one-hot genome")
        self.__data = one_hot_ann_genome.data
        self.__seq_info_genome = SeqInfoContainer()
        self.__ANN_TYPES=one_hot_ann_genome.ANN_TYPES
        self.__parse_seqs()
    @property
    def regions(self):
        return self.__seq_info_genome
    def __parse_seqs(self):
        for seq in self.__seq_info_genome.data:
            self.__parse_regions(seq)
    def __parse_regions(self,seq):
        for ann_type in self.__ANN_TYPES:
            self.__parse_of_region(ann_type, seq.get_annotation(ann_type))
    def __parse_of_region(self,ann_type, ann_seq):
        copied_ann_seq = ann_seq.copy()
        region_id_prefix = 'region_'
        region_id = 0
        length = len(copied_ann_seq)
        if length%2!=0:
            copied_ann_seq.append('X')
        copied_ann_seq.shape=(length/2,2)
        target = SeqInformation()
        
        target.ann_type = ann_type
        for index, sub_seq in enumerate(copied_ann_seq):
            if index==0:
                if np.all(sub_seq==[1,1]):
                    target.start = index
            else:
                if np.all(sub_seq==[0,1]):
                    target.start = index
            if np.all(sub_seq==[1,0]) or np.all(sub_seq==[1,'X']):
                if target.start is not None:
                    target.end = index
                else:
                    raise Exception("Annotations is incomplete")
            if target.start is not None and target.end is not None:
                target.ann_type = ann_type
                target.ann_status = "whole"
                target.id = region_id_prefix+str(region_id)
                self.__seq_info_genome.add(target)
                region_id+=1
                target = SeqInformation()
                
