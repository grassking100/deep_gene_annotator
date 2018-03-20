from . import FastaConverter
from . import LengthNotEqualException
import os
import numpy as np
import keras.backend as K
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from keras.preprocessing.sequence import pad_sequences
class DataHandler(metaclass=ABCMeta):
    @abstractmethod
    def get_data(self,data_path,answer_path):
        pass
class SeqAnnDataHandler(DataHandler):
    @staticmethod
    def to_vecs(data_pair_list):
        inputs = []
        answers = []
        for data_pair in data_pair_list:
            inputs.append(data_pair['input'])
            answers.append(data_pair['answer'])
        inputs = np.array(inputs)
        answers = np.array(answers)
        return (inputs,answers)
    @staticmethod
    def padding(inputs, answers, padding_signal):
        align_inputs = pad_sequences(inputs, padding='post',value=0)
        align_answers = pad_sequences(answers, padding='post',value=padding_signal)
        return (align_inputs, align_answers)
    @staticmethod
    def get_seq_vecs(fasta_path,discard_invalid_seq):
        fasta_path = os.path.abspath(fasta_path)
        fasta_converter = FastaConverter()
        seq_dict = fasta_converter.to_seq_dict(fasta_path)
        return fasta_converter.to_vec_dict(seq_dict = seq_dict,discard_invalid_seq=discard_invalid_seq)
    @staticmethod
    def get_ann_vecs(path,ann_types):
        path = os.path.abspath(path)
        ann_count = {}
        ann_data = np.load(path).item()
        dict_ = {}
        for name,data in ann_data.items():
            ann = []
            for type_ in ann_types:
                value = data[type_]
                if type_ not in ann_count.keys():
                    ann_count[type_] = 0
                ann_count[type_] += np.sum(value)
                ann.append(value)
            dict_[name] = np.transpose(ann)
        return dict_,ann_count
    @staticmethod
    def process_tensor(answer, prediction,values_to_ignore=None):
        """Remove specific ignored singal and concatenate them into a two dimension tensor"""
        reshape_prediction = K.reshape(prediction, [-1])
        reshape_answer = K.reshape(answer, [-1])
        if values_to_ignore is not None:
            if not isinstance(values_to_ignore,list):
                values_to_ignore=[values_to_ignore]
            index = tf.where(K.not_equal(reshape_answer, values_to_ignore))
            clean_prediction = K.reshape(K.gather(reshape_prediction, index),
                                         [-1, K.shape(prediction)[2]])
            clean_answer = K.reshape(K.gather(reshape_answer, index),
                                     [-1, K.shape(answer)[2]])
        else:
            clean_prediction = K.reshape(prediction, [-1, K.shape(prediction)[2]])
            clean_answer = K.reshape(answer, [-1, K.shape(answer)[2]])
        return (clean_answer, clean_prediction)
    @staticmethod
    def _to_dict(seqs,answer):
        dict_ = {'data_pair':{},'annotation_count':answer[1]}
        ann_vecs = answer[0]
        for name,seq in seqs.items():
            ann_vec = ann_vecs[name]
            ann_length = np.shape(ann_vec)[0]
            seq_length = np.shape(seq)[0]
            if ann_length != seq_length:
                raise LengthNotEqualException(ann_length, seq_length)
            dict_['data_pair'][name]={'input':seq,'answer':ann_vec}
        return dict_
    @classmethod
    def get_data(cls,seq_path,answer_path,ann_types,discard_invalid_seq):
        seqs = cls.get_seq_vecs(seq_path,discard_invalid_seq)
        answer = cls.get_ann_vecs(answer_path,ann_types)
        return cls._to_dict(seqs,answer)