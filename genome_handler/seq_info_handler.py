from . import numpy as np
from . import matplotlib_pyplot as plt
from . import random
from . import csv
from . import re
from . import deepdish
#Purpose:Get selected sequence information by sequence selected principle
#Input:Region information list,Sequence selected principle and genome information about chromosomes' id and its length
#Output:Selected Sequence's start index,end index,and other information
class SeqInfoGenerator:
    def __init__(self,half_length,max_diff,chromosomes_info,seed_id_prefix,seq_id_prefix):
        self.__chromosomes_info=chromosomes_info
        self.__seed_id_prefix=seed_id_prefix
        self.__seq_id_prefix=seq_id_prefix
        self.__half_length=half_length
        self.__max_diff=max_diff
        self.__selected_info_seeds=[]
        self.__seed_id=0
        self.__seq_id=0
    def __get_range(self,index,min_value,max_value):
        new_start=index-(self.__half_length+random.randint(0,self.__max_diff))
        new_end=index+(self.__half_length+random.randint(0,self.__max_diff))
        new_start=self.__safe_value(new_start,min_value,max_value)
        new_end=self.__safe_value(new_end,min_value,max_value)
        return {'start':new_start,'end':new_end}
    def __create_seed_info(self,region_information,annotation_type):
        self.__seed_id+=1
        mode_text=['start','end','middle']
        mode=[region_information['start'],region_information['end'],int((region_information['start']+region_information['end'])/2)]
        mode_selected_index=random.randint(0,len(mode)-1)
        seed_info={"chrom":region_information['chrom'],
                   "strand":region_information['strand'],
                   "id":self.__seed_id_prefix+"_"+str(self.__seed_id),
                   "information":annotation_type,
                   "mode":mode_text[mode_selected_index],
                   "middle index":mode[mode_selected_index]
                  }
        return seed_info
    def __create_selected_sequences_from_seed(self,seed_info):
        self.__seq_id+=1
        sequence_info=self.__get_range(seed_info['middle index'],0,self.__chromosomes_info[seed_info['chrom']]-1)
        sequence_info['information']=seed_info['information']
        sequence_info['mode']=seed_info['mode']
        sequence_info['chrom']=seed_info['chrom']
        sequence_info['feature']=seed_info['id']
        sequence_info['strand']=seed_info['strand']
        sequence_info['id']=self.__seq_id_prefix+"_"+str(self.__seq_id)
        return sequence_info
    def create_selected_sequences_info(self,region_information_list,duplicate_number,annotation_type):
        sequences_info=[]
        for region_information in region_information_list:
            seed_info=self.__create_seed_info(region_information,annotation_type)
            self.__selected_info_seeds.append(seed_info)
            for i in range(0,duplicate_number):
                temp=self.__create_selected_sequences_from_seed(seed_info)
                sequences_info.append(temp)
        return sequences_info
    def __safe_value(self,value,min_value,max_value):
        if value<min_value:
            return min_value
        if value>max_value:
            return max_value
        return value
    @property
    def selected_info_seeds(self):
        return self.__selected_info_seeds
    
#Purpose:Get selected sequences information by sequence selected principle
#Input:A dictionary stored array of annotated region information,sequence selected principle and genome information about chromosomes' id and its length
#Output:Selected Sequences start index,end index,and other information
class SeqInfoExtractor:
    def __init__(self,regionExtractor,select_principle,chromosomes_info):
        self.__data=regionExtractor.regions
        self.__selected_info_seeds=[]
        self.__principle=select_principle
        self.__sequences_info=[]
        self.__chromosomes_info=chromosomes_info
        self.__init_seq_info_generator()
        self.__create_sequences_info()
    @property
    def sequences_info(self):
        return self.__sequences_info
    def __init_seq_info_generator(self): self.__generator=SeqInfoGenerator(self.__principle['region_half_length'],self.__principle['max_shift_length_diff'],self.__chromosomes_info,self.__principle['seed_id_prefix'],self.__principle['seq_id_prefix'])
    def __create_sequences_info(self):
        ann_types=['cds','utr_5','utr_3','intron','intergenic_region']
        for ann_type in ann_types:
            data=self.__data[ann_type]
            if self.__principle['remove_end_of_strand']:
                used_data=self.__get_clean_data(data)
            else:
                used_data=data
            if self.__principle['used_all_data_without_random_choose']:
                selected_data=used_data
            else:
                number=self.__principle['selected_target_settings'][ann_type]
                selected_data=np.random.choice(used_data,number,replace=True) 
            self.__sequences_info+=self.__generator.create_selected_sequences_info(selected_data,self.__principle['sample_number_per_region'],ann_type)
        self.__selected_info_seeds=self.__generator.selected_info_seeds
    def __get_clean_data(self,data):
        clean_data=[]
        for item in data:
            chrom_length=self.__chromosomes_info[item['chrom']]
            if not (item['start']==0 or item['end']==0 or item['start']==chrom_length-1 or item['end']==chrom_length-1):
                clean_data.append(item)
            else:
                print("Following seed won't be used to generate data")
                print(item)
        return clean_data
    def save_selected_info_seeds(self,path,notes):
        with open(path, 'w') as f:
            w = csv.writer(f, delimiter=',')
            for note in notes:
                w.writerow(["##"+note])
            w.writerow(["id","annotation type","mode","index","chrom","strand",])
            for seed in self.__selected_info_seeds:
                w.writerow([seed['id'],seed['information'],seed['mode'],seed['middle index'],seed['chrom'],seed['strand']])
    def save_as_GTF(self,path,source,notes):
        with open(path, 'w') as f:
            w = csv.writer(f, delimiter='\t')
            for note in notes:
                w.writerow(["##"+note])
            for seq in self.sequences_info:
                attribute=""
                attribute+="seed_id="+str(seq['feature'])+";"
                attribute+="annotation_type="+str(seq['information'])+";"
                attribute+="mode="+str(seq['mode'])+";"
                w.writerow([seq['chrom'],source,seq['id'],seq['start']+1,seq['end']+1,'.',seq['strand'],'.','.',attribute])
