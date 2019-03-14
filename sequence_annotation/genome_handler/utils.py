import math
import numpy as np
import pandas as pd
import random
from . import ann_seq_processor
from ..utils.exception import LengthNotEqualException
from .sequence import AnnSequence
from .seq_container import AnnSeqContainer

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

def preprocess_ensembl_data(parsed_file_path,valid_chroms_id,
                            merged_by='Protein stable ID',gene_types=None):
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

def ann_count(ann_seqs):
    ann_count = {}
    types = ann_seqs.ANN_TYPES
    for seq in ann_seqs:
        for type_ in types:
            if type_ not in ann_count.keys():
                ann_count[type_] = 0
            ann_count[type_] += np.sum(seq.get_ann(type_))
    return ann_count

def loader(fasta,ann_seqs,min_len=None,max_len=None,ratio=None,outlier_coef=1.5,simplify_map=None,num_max=None):
    seqs_len = [len(seq) for seq in ann_seqs]
    seqs_len.sort()
    min_len = min_len or 0
    outlier_name = None
    if max_len is None:
        ratio = ratio or 0.01
        max_len = seqs_len[:int(len(seqs_len)*ratio)][-1]
    inner_fasta = {}
    for seq in ann_seqs:
        if min_len <= len(seq) <= max_len:
            inner_fasta[seq.id]=fasta[seq.id]
        elif len(seq) >= max_len*outlier_coef:
            outlier_name = seq.id
    keys = list(inner_fasta.keys())
    random.shuffle(keys)
    if num_max is not None:
        keys = keys[:num_max]
    selected_seqs = AnnSeqContainer()
    if simplify_map is not None:
        selected_seqs.ANN_TYPES = list(simplify_map.keys())
    else:
        selected_seqs.ANN_TYPES = ann_seqs.ANN_TYPES
    selected_fasta = {}
    number = 0
    for seq_id in keys:
        seq = ann_seqs.get(seq_id)
        if simplify_map is not None:
            seq = ann_seq_processor.mixed_typed_seq_generate(seq)
            seq = ann_seq_processor.simplify_seq(seq,simplify_map)
        selected_seqs.add(seq)
        selected_fasta[seq_id]=inner_fasta[seq_id]
        number += 1
    outlier_seq = None
    outlier_fasta = None
    if outlier_name is not None:
        outlier_seq = ann_seqs.get(outlier_name)
        outlier_fasta = fasta[outlier_name]
        if simplify_map is not None:
            outlier_seq = ann_seq_processor.mixed_typed_seq_generate(outlier_seq)
            outlier_seq = ann_seq_processor.simplify_seq(outlier_seq,simplify_map)
    return selected_fasta,selected_seqs,outlier_fasta,outlier_seq

def get_subseqs(ids,ann_seqs):
    sub_seqs = AnnSeqContainer()
    sub_seqs.ANN_TYPES = ann_seqs.ANN_TYPES
    sub_seqs.add([ann_seqs[id_] for id_ in ids])
    return sub_seqs