import math
import numpy as np
import pandas as pd
from . import AnnSeqContainer
def merge_data(data,groupby,raise_duplicated_excpetion=True):
    def sort_aggregate(x):
        value_list = []
        for item in x.tolist():
            if isinstance(item,str):
                value_list.append(item)
            elif not math.isnan(item):
                value_list.append(item)
        value_set = set(value_list)
        if len(value_set)==1:
            return value_list[0]
        else:
            if len(value_list)==len(value_set):
                return value_list
            else:
                if raise_duplicated_excpetion:
                    raise Exception("There is duplicated data"+str(value_list))
                else:
                    return value_list
    result = data.groupby(groupby).agg(sort_aggregate)
    result.reset_index(inplace=True)
    if set(data.columns)==set(result.columns):
        return result
    else:
        raise Exception("Columns is not the same")
        
def gene_boundary(data,raise_duplicated_excpetion=True):
    def sort_aggregate(x):
        value_list = []
        for item in x.tolist():
            if isinstance(item,str):
                value_list.append(item)
            elif not math.isnan(item):
                value_list.append(item)
        value_set = set(value_list)
        if len(value_set)==1:
            return value_list[0]
        else:
            if len(value_list)==len(value_set):
                return value_list
            else:
                if raise_duplicated_excpetion:
                    raise Exception("There is duplicated data"+str(value_list))
                else:
                    return value_list
    def get_min(data):
        set_ = set(np.hstack(data.tolist()))
        return min(set_)
    def get_max(data):
        set_ = set(np.hstack(data.tolist()))
        return max(set_)
    basic_data = data[['Gene stable ID','Transcript stable ID',
                       'Protein stable ID','Chromosome/scaffold name',
                       'Strand']].groupby('Gene stable ID').agg(sort_aggregate)
    starts = data[['Gene stable ID','Transcript start (bp)']].groupby('Gene stable ID').agg(get_min)
    end = data[['Gene stable ID','Transcript end (bp)']].groupby('Gene stable ID').agg(get_max)
    result = pd.concat([basic_data,end,starts], axis=1)
    result.reset_index(inplace=True)
    return result

def annotation_count(ann_seqs):
    count = {}
    for ann_seq in ann_seqs:
        for type_ in ann_seq.ANN_TYPES:
            if type_ not in count.keys():
                count[type_] = 0
            count[type_] += np.sum(ann_seq.get_ann(type_))
    return count

def preprocess_ensembl_data(parsed_file_path,valid_chroms_id,merged_by='Protein stable ID'):
    file = pd.read_csv(parsed_file_path,sep='\t',dtype={'Gene stable ID':np.str ,
                                                        'Protein stable ID':np.str ,
                                                        'Transcript stable ID':np.str ,
                                                        'Chromosome/scaffold name':np.str })
    valid_data = all_[all_['Chromosome/scaffold name'].isin([str(char) for char in valid_chroms_id])]
    valid_data = valid_data.drop_duplicates()
    merged_data = merge_data(valid_data,merged_by)
    return merged_data

def simplify_genome(genome,mapping_dictionary):
    simplified = AnnSeqContainer()
    simplified.ANN_TYPES = list(set(mapping_dictionary.values()))
    for seq in genome:
        ann_seq = AnnSequence()
        ann_seq.from_dict(seq.to_dict())
        ann_seq.clean_space()
        ann_seq.ANN_TYPES = simplified.ANN_TYPES
        ann_seq.init_space()
        for source_type,target_domain in mapping_dictionary.items():
            ann_seq.add_ann(target_domain,seq.get_ann(source_type))
        simplified.add(ann_seq)
    return simplified
