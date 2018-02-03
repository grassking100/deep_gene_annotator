import numpy as np
import random
from . import SeqInformation
class SeqInfoGenerator:
    """Extract selected regions' annotation information"""
    def __init__(self, region_container, principle,
                 chromosomes_info, seed_id_prefix, seq_id_prefix):
        self.__data=self._to_tree(region_container)
        self.__principle=principle
        self.__chromosomes_info=chromosomes_info
        self.__selected_info_seeds=[]
        self.__sequences_info=[]
        self.__create_sequences_info()
        self.__seed_id = 0
        self.__seq_id = 0
        self.__seed_id_prefix = seed_id_prefix
        self.__seq_id_prefix = seq_id_prefix
    def _to_tree(self,region_container):
        data = region_container.data()
        tree = {}
        for seq_info in data:
            key = seq_info.ann_type + "_" + seq_info.seq_ann_status
            if key not in tree.keys():
                tree[key] = []
            tree[key].append(seq_info)
        return tree
    @property
    def sequences_info(self):
        return self.__sequences_info
    def __get_selected_regions(self):
        for ann_type in self.__data.keys():
            data = self.__data[ann_type]
            if self.__principle['remove_end_of_strand']:
                used_data=self.__get_clean_data(data)
            else:
                used_data=data
            if self.__principle['used_all_data_without_random_choose']:
                selected_region=used_data
            else:
                number=self.__principle['selected_target_settings'][ann_type]
                selected_region=np.random.choice(used_data,number,replace=True) 
            return selected_region
    def __create_sequences_info(self):
        regions = self.__get_selected_regions()
        number = self.__principle['sample_number_per_region']
        for ann_type in self.__data.keys():
            self.__sequences_info += [self._create_selected_sequences_info(regions,number,ann_type)]
    def __get_clean_data(self,data):
        clean_data=[]
        for item in data:
            chrom_length=self.__chromosomes_info[item['chrom']]
            if not (item['start']==0 or item['end']==0 or
                    item['start']==chrom_length-1 or
                    item['end']==chrom_length-1):
                clean_data.append(item)
            else:
                print("Following seed won't be used to generate data")
                print(item)
        return clean_data
    def __create_seed_info(self,region_info, ann_type):
        mode_text=['start','end','middle']
        mode=[region_info['start'],region_info['end'],
              int((region_info['start']+region_info['end'])/2)]
        mode_selected_index=random.randint(0,len(mode)-1)
        seed_info = SeqInformation()
        seed_info.id = self.__seed_id_prefix+"_"+str(self.__seed_id)
        seed_info.ann_type = ann_type
        seed_info.ann_status = mode_text[mode_selected_index]
        seed_info.extra_index = mode[mode_selected_index]
        seed_info.extra_index_name = mode_text[mode_selected_index]
        self.__seed_id+=1
        return seed_info
    def __create_seq_info(self,seed_info, length):
        sequence_info = seed_info.copy()
        half_length = self.__principle['half_length']
        max_diff = self.__principle['max_diff']
        index = sequence_info.extra_index
        new_start=index-(half_length+random.randint(0,max_diff))
        new_end=index+(half_length+random.randint(0,max_diff))
        if new_start >= length:
            new_start = length -1
        if new_end < 0:
            new_start = 0
        sequence_info.start =new_start
        sequence_info.end = new_end
        sequence_info.id=self.__seq_id_prefix+"_"+str(self.__seq_id)
        self.__seq_id += 1
        return sequence_info
    def _create_selected_sequences_info(self,region_info_list,number,ann_type):
        sequences_info=[]
        for region_info in region_info_list:
            seed_info=self.__create_seed_info(region_info,ann_type)
            self.__selected_info_seeds.append(seed_info)
            length = self.__chromosomes_info[region_info.chromosome_id]
            for index in range(number):
                temp=self.__create_seq_info(seed_info,length)
                sequences_info.append(temp)
        return sequences_info
