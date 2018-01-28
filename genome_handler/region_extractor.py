from . import GenomeAnnotator
from . import numpy as np
from . import re
from . import csv
#Purpose:Get annotated region information
#Input:A Tree structure made from dictionary,its node stored sequence of numeric value
#Output:A dictionary stored array of annotated region information
class RegionExtractor:
    def __init__(self,genomeAnnotator,is_normalized):
        self.__data=genomeAnnotator.get_genome(is_normalized)
        self.__regions={}
        self.__ANNOTATION_TYPES=['intergenic_region','utr_3','utr_5','intron','cds']
        for annotation_type in self.__ANNOTATION_TYPES:
            self.__regions[annotation_type]=[]
        self.__record_regions()
    def __record_regions(self):
        data=GenomeAnnotator.flatten_data(self.__data)
        for sequence_info_and_sequence in data:
            regions=self.__get_regions_of_seq_info(sequence_info_and_sequence['sequence'])
            temp=self.__add_info_for_every_region(regions,sequence_info_and_sequence['chromosome_id'],sequence_info_and_sequence['strand'])
            self.__regions[sequence_info_and_sequence['annotation_type']]+=temp
    def __add_info_for_every_region(self,regions,chrom,strand):
        new_regions=[]
        for region in regions:
            region['chrom']=chrom
            region['strand']=strand
            new_regions.append(region)
        return new_regions
    def __get_regions_of_seq_info(self,normalized_sequence):
        ann_seq_str_arr=np.ceil(normalized_sequence).astype(int).astype(str)
        ann_seq_str="".join(ann_seq_str_arr)
        starts=[]
        ends=[]
        if ann_seq_str_arr[0]=='1':
            starts.append(0)
        for start in re.finditer('(01)',ann_seq_str):
            starts.append(start.start()+1)
        for end in re.finditer('10',ann_seq_str):
            ends.append(end.start())
        if ann_seq_str_arr[-1]=='1':
            ends.append(len(ann_seq_str_arr)-1)
        if len(starts)!=len(ends):
            raise Exception("Annotations is incomplete")
        else:
            data=[]
            for start,end in zip(starts,ends):
                data.append({"start":start,"end":end})
        return data
    @property
    def regions(self):
        return self.__regions
    def save_as_GTF(self,path,source):
        with open(path, 'w') as f: 
            w = csv.writer(f, delimiter='\t')
            for annotation_type in self.__ANNOTATION_TYPES:
                seqs=self.__regions[annotation_type]
                for seq in seqs:
                    w.writerow([seq['chrom'],source,annotation_type,seq['start']+1,seq['end']+1,'.',seq['strand'],'.','.'])