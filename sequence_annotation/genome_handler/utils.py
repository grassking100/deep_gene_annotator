import math
import numpy as np
import pandas as pd
from ..utils.exception import LengthNotEqualException

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

def preprocess_ensembl_data(parsed_file_path,valid_chroms_id,merged_by='Protein stable ID',gene_types=None):
    gene_types = gene_types or ['protein_coding']
    file = pd.read_csv(parsed_file_path,sep='\t',dtype={'Gene stable ID':np.str ,
                                                        'Protein stable ID':np.str ,
                                                        'Transcript stable ID':np.str ,
                                                        'Chromosome/scaffold name':np.str })
    gene_type_status = file['Gene type'].isin(gene_types)
    chrom_status = file['Chromosome/scaffold name'].isin([str(char) for char in valid_chroms_id])
    valid_data = file[gene_type_status & chrom_status]
    valid_data = valid_data.drop_duplicates()
    merged_data = merge_data(valid_data,merged_by)
    return merged_data

def to_dict(seqs,answer,ann_types):
    ann_count = {}
    data_pair = {}
    ann_vecs = answer
    for name,seq in seqs.items():
        ann_vec = ann_vecs[str(name)]
        transposed_ann_vec = np.transpose(ann_vec)
        for index, type_ in enumerate(ann_types):
            if type_ not in ann_count.keys():
                ann_count[type_] = 0
            ann_count[type_] += np.sum(transposed_ann_vec[index])
        ann_length = np.shape(ann_vec)[0]
        seq_length = np.shape(seq)[0]
        if ann_length != seq_length:
            raise LengthNotEqualException(ann_length, seq_length)
        data_pair[name]={'input':seq,'answer':ann_vec}
    dict_ = {'data_pair':data_pair,'annotation_count':ann_count}
    return dict_
