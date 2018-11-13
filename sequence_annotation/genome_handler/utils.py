import keras.backend as K
import tensorflow as tf
import math
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from .seq_container import AnnSeqContainer
from ..data_handler.seq_converter import SeqConverter
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

def preprocess_ensembl_data(parsed_file_path,valid_chroms_id,merged_by='Protein stable ID',gene_types=['protein_coding']):
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

def process_tensor(answer, prediction,values_to_ignore=None):
    """Remove specific ignored singal and concatenate them into a two dimension tensor"""
    true_shape = K.int_shape(answer)
    pred_shape = K.int_shape(prediction)
    if len(true_shape)!=3 or len(pred_shape)!=3:
        raise Exception("Shape must be 3 dimensions")
    true_label_number = K.shape(answer)[-1]
    pred_label_number = K.shape(prediction)[-1]
    reshape_prediction = K.reshape(prediction, [-1])
    reshape_answer = K.reshape(answer, [-1])
    if values_to_ignore is not None:
        if not isinstance(values_to_ignore,list):
            values_to_ignore=[values_to_ignore]
        index = tf.where(K.not_equal(reshape_answer, values_to_ignore))
        clean_prediction = K.reshape(K.gather(reshape_prediction, index),
                                     [-1, pred_label_number])
        clean_answer = K.reshape(K.gather(reshape_answer, index),
                                 [-1, true_label_number])
    else:
        clean_prediction = K.reshape(prediction, [-1,pred_label_number])
        clean_answer = K.reshape(answer, [-1, true_label_number])
    true_shape = K.int_shape(clean_answer)
    pred_shape = K.int_shape(clean_prediction)
    if true_shape != pred_shape:
        raise LengthNotEqualException(true_shape, pred_shape)
    return (clean_answer, clean_prediction)

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

def padding(inputs, answers, padding_signal):
    align_inputs = pad_sequences(inputs, padding='post',value=0)
    align_answers = pad_sequences(answers, padding='post',value=padding_signal)
    return (align_inputs, align_answers)