"""This submodule provides library about visualize"""
import matplotlib.pyplot as plt
import numpy as np
from sequence_annotation.genome_handler.sequence import STRANDS, PLUS
from sequence_annotation.genome_handler.exception import InvalidStrandType

def visual_ann_seq(seq):
    """Visualize the count of each type along sequence"""
    answer_vec = []
    for type_ in seq.ANN_TYPES:
        answer_vec.append(np.array([0.0]*seq.length))
    for index,type_ in enumerate(seq.ANN_TYPES):
        if seq.strand not in STRANDS:
            raise InvalidStrandType(seq.strand)
        if seq.strand==PLUS:
            answer_vec[index] += seq.get_ann(type_)
        else:
            answer_vec[index] += np.flip(seq.get_ann(type_),0)
            
    x = list(range(seq.length))
    plt.stackplot(x,answer_vec,labels=seq.ANN_TYPES)
    plt.legend(loc='upper right')

def visual_ann_genome(seqs):
    """Visualize the count of each type along sequences"""
    answer_vec = []
    max_len = 0
    for seq in seqs:
        max_len = max(seq.length,max_len)
    for type_ in seqs.ANN_TYPES:
        answer_vec.append(np.array([0.0]*max_len))
    for seq in seqs:
        for index,type_ in enumerate(seqs.ANN_TYPES):
            if seq.strand not in STRANDS:
                raise InvalidStrandType(seq.strand)
            if seq.strand==PLUS:
                answer_vec[index][0:seq.length] += seq.get_ann(type_)
            else:
                answer_vec[index][0:seq.length] += np.flip(seq.get_ann(type_),0)
    x = list(range(np.array(answer_vec).shape[1]))
    plt.stackplot(x,answer_vec, labels=seqs.ANN_TYPES)
    plt.legend(loc='upper right')

def position_error(predict, answer,check_length=True):
    """Calculate the error of each type along sequence"""
    if predict.ANN_TYPES != answer.ANN_TYPES:
        err = "Predicted sequence and answer should have same annotation type, get {} and {}."
        raise Exception(err.format(predict.ANN_TYPES,answer.ANN_TYPES))

    if check_length and len(predict)!=len(answer):
        err = "Prediction and answer should have same length, get {} and {}."
        raise Exception(err.format(len(predict),len(answer)))

    if predict.strand != answer.strand:
        err = "Prediction and answer should have same strand, get {} and {}."
        raise Exception(err.format(predict.strand,answer.strand))

    if predict.chromosome_id != answer.chromosome_id:
        err = "Prediction and answer should have same chromosome id, get {} and {}."
        raise Exception(err.format(predict.chromosome_id,answer.chromosome_id))

    error = {}
    for type_ in predict.ANN_TYPES:
        status = np.zeros(max(len(predict),len(answer)),dtype=np.float32)
        status[:len(predict)] = predict.get_ann(type_)
        status[:len(answer)] -= answer.get_ann(type_)
        if predict.strand not in STRANDS:
            raise InvalidStrandType(predict.strand)
        if predict.strand != PLUS:
            status = np.flip(status)
        error[type_] = status
    return  error

def visual_error(predict, answer,check_length=True):
    """Visualize the error of each type along sequence"""
    error_status = position_error(predict, answer,check_length=check_length)
    for type_ in predict.ANN_TYPES:
        error = error_status[type_]
        plt.plot(error, label=type_)
    plt.legend(loc='upper right')
    
def partial_dependence_plot(df,column_name,
                            round_value=None,only_show_completed=True,
                             xlabel=None,ylabel=None,title=None):
    if only_show_completed:
        df=df[df['state']=='COMPLETE']
    round_value = round_value or 1
    group = df.groupby(column_name)['value']
    value = dict(group.mean())
    median = dict(group.median())
    std = dict(group.std())
    max_ = dict(group.max())
    min_ = dict(group.min())
    sorted_values = []
    sorted_medians= []
    sorted_stds = []
    sorted_maxs = []
    sorted_mins = []
    sorted_keys = sorted(list(value.keys()))
    
    for column_name in sorted_keys:
        sorted_values.append(round(100*value[column_name],round_value))
        sorted_medians.append(round(100*median[column_name],round_value))
        sorted_stds.append(round(100*std[column_name],round_value))
        sorted_maxs.append(round(100*max_[column_name],round_value))
        sorted_mins.append(round(100*min_[column_name],round_value))

    sorted_medians = np.array(sorted_medians)
    sorted_values = np.array(sorted_values)
    sorted_stds = np.array(sorted_stds)
    sorted_maxs = np.array(sorted_maxs)
    sorted_mins = np.array(sorted_mins)

    if xlabel is not None:
        plt.xlabel(xlabel)
    
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.plot(sorted_keys,sorted_medians,label='median')
    plt.plot(sorted_keys,sorted_values,label='mean')
    plt.fill_between(sorted_keys,sorted_values-sorted_stds,sorted_values+sorted_stds,alpha=.4,label='std')
    plt.fill_between(sorted_keys,sorted_mins,sorted_maxs,alpha=.2,label='boundary')
    plt.legend()
