import os
import errno
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import math

def logical_not(lhs, rhs):
    return np.logical_and(lhs,np.logical_not(rhs))

def create_folder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as erro:
        if erro.errno != errno.EEXIST:
            raise

def get_protected_attrs_names(object_):
    class_name = object_.__class__.__name__
    attrs = [attr for attr in dir(object_) if attr.startswith('_') 
             and not attr.endswith('__')
             and not attr.startswith('_'+class_name+'__')]
    return attrs

def reverse_weights(cls, class_counts, epsilon=1):
    scale = len(class_counts.keys())
    raw_weights = {}
    weights = {}
    for type_,count in class_counts.items():
        if count > 0:
            weight = 1 / count
        else:
            if epsilon > 0:
                weight = 1 / (count+epsilon)
            else:
                raise Exception(type_+" has zero count,so it cannot get reversed count weight")
        raw_weights[type_] = weight
    sum_raw_weights = sum(raw_weights.values())
    for type_,weight in raw_weights.items():
        weights[type_] = scale * weight / (sum_raw_weights)
    return weights

def process_tensor(answer, prediction,values_to_ignore=None):
    """Remove specific ignored singal and concatenate them into a two dimension tensors"""
    true_shape = K.int_shape(answer)
    pred_shape = K.int_shape(prediction)
    if len(true_shape)!=3 or len(pred_shape)!=3:
        raise Exception("Shape must be 3 dimensions")
    if values_to_ignore is not None:
        if not isinstance(values_to_ignore,list):
            values_to_ignore=[values_to_ignore]
        index = tf.where(K.any(K.not_equal(answer,values_to_ignore),-1))
        clean_answer = tf.gather_nd(answer, index)
        clean_prediction = tf.gather_nd(prediction, index)
    else:
        true_label_number = K.shape(answer)[-1]
        pred_label_number = K.shape(prediction)[-1]
        clean_answer = K.reshape(answer, [-1, true_label_number])
        clean_prediction = K.reshape(prediction, [-1,pred_label_number])
    true_shape = K.int_shape(clean_answer)
    pred_shape = K.int_shape(clean_prediction)
    if true_shape != pred_shape:
        raise LengthNotEqualException(true_shape, pred_shape)
    return (clean_answer, clean_prediction)

def padding(inputs, answers, padding_signal):
    align_inputs = pad_sequences(inputs, padding='post',value=0)
    align_answers = pad_sequences(answers, padding='post',value=padding_signal)
    return (align_inputs, align_answers)

def model_method(model,input_index,output_index):
    return K.function([model.layers[input_index].input], [model.layers[output_index].output])

def split(ids,ratios):
    if round(sum(ratios))!=1:
        raise Exception("Ratio sum should be one")
    lb = ub = 0
    ids_list=[]
    id_len = len(ids)
    sum_=0
    for index in range(len(ratios)):
        ub += ratios[index]
        item = ids[math.ceil(lb*id_len):math.ceil(ub*id_len)]
        sum_+=len(item)
        ids_list.append(item)
        lb=ub
    if sum_!=id_len:
        raise Exception("Id number is not consist with origin count")
    return ids_list

def index2onehot(index,channel_size):
    if (np.array(index)<0).any() or (np.array(index)>=channel_size).any():
        raise Exception("Invalid number")
    L = len(index)
    loc = list(range(L))
    onehot = np.zeros((channel_size,L))
    onehot[index,loc]=1
    return onehot

def get_subdict(ids,data):
    return dict(zip(ids,[data[id_] for id_ in ids]))