import math
import random
import numpy as np
import pandas as pd
from .seq_container import AnnSeqContainer,SeqInfoContainer
from .region_extractor import GeneInfoExtractor
from .ann_seq_processor import vecs2seq

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
    count = {}
    types = ann_seqs.ANN_TYPES
    for seq in ann_seqs:
        for type_ in types:
            if type_ not in count.keys():
                count[type_] = 0
            count[type_] += np.sum(seq.get_ann(type_))
    return count

def get_subseqs(ids,ann_seqs):
    sub_seqs = AnnSeqContainer(ann_seqs.ANN_TYPES)
    sub_seqs.add([ann_seqs[id_] for id_ in ids])
    return sub_seqs

def index2onehot(index,channel_size):
    if (np.array(index)<0).any() or (np.array(index)>=channel_size).any():
        raise Exception("Invalid number")
    L = len(index)
    loc = list(range(L))
    onehot = np.zeros((channel_size,L))
    onehot[index,loc]=1
    return onehot

def ann2onehot(ann_seq,length=None):
    C,L = ann_seq.shape
    index = ann_seq.argmax(0)
    if length is not None:
        index = index[:length]
    return index2onehot(index,C)

def ann2seq_info(ids,ann_seqs,lengths,ann_types,simplify_map,extractor=None):
    if extractor is None:
        extractor = GeneInfoExtractor()
    seq_infos = SeqInfoContainer()
    N,C,L = ann_seqs.shape
    for id_,output, length in zip(ids,ann_seqs,lengths):
        output = ann2onehot(output,length)
        seq = vecs2seq(output,id_,'plus',ann_types)
        infos = extractor.extract_per_seq(seq,simplify_map)
        seq_infos.add(infos)
    return seq_infos
