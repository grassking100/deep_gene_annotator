from . import FastaConverter
from . import SeqConverter
from . import LengthNotEqualException
from . import InvalidStrandType
from . import AnnSeqContainer
from . import annotation_count
import os
import warnings
import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from keras.preprocessing.sequence import pad_sequences
from os.path import expanduser,abspath
class DataHandler(metaclass=ABCMeta):
    @classmethod
    def get_weight(cls,class_counts, method_name):
        if method_name=="reversed_count_weight":
            return cls._reversed_count_weight(class_counts)
        else:
            mess = method_name+" is not implement yet."
            raise Exception(mess)
    @classmethod
    def _reversed_count_weight(cls,class_counts):
        scale = len(class_counts.keys())
        raw_weights = {}
        weights = {}
        for type_,count in class_counts.items():
            if count > 0:
                weight = 1 / count
            else:
                raise Exception(type_+" has zero count,so it cannot get reversed count weight")
            raw_weights[type_] = weight
        sum_raw_weights = sum(raw_weights.values())
        for type_,weight in raw_weights.items():
            weights[type_] = scale * weight / (sum_raw_weights )
        return weights
    @abstractmethod
    def get_data(self,data_path,answer_path,ann_types,setting=None):
        pass
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
class SimpleDataHandler(DataHandler):
    @staticmethod
    def _to_dict(inputs_,answer):
        dict_ = {'data_pair':{},'cell_types':answer[1]}
        answer_vecs = answer[0]
        for name,input_ in inputs_.items():
            answer_vec = answer_vecs[name]
            dict_['data_pair'][name]={'input':input_,'answer':answer_vec}
        return dict_
    @classmethod
    def get_data(cls,data_path,answer_path,ann_types,setting=None):
        bulk_expression = cls.get_bulk_expression(abspath(expanduser(data_path)))
        proportion = cls.get_proportion(abspath(expanduser(answer_path)))
        return cls._to_dict(bulk_expression,proportion)
    @staticmethod
    def get_bulk_expression(path):
        raw_data = pd.read_csv(path,sep='\t',index_col=0)
        bulk_ids = list(raw_data)
        data = {}
        for id_ in bulk_ids:
            data[id_] = list(raw_data[id_])[:-1]
        return data
    @staticmethod
    def get_proportion(path):
        raw_data = pd.read_csv(path,sep='\t',index_col=0)
        cell_types = list(raw_data.index)[:-1]
        bulk_ids = list(raw_data)
        data = {}
        for id_ in bulk_ids:
            data[id_] = list(raw_data[id_])[:-1]
        return (data,cell_types)
class SeqAnnDataHandler(DataHandler):
    @staticmethod
    def padding(inputs, answers, padding_signal):
        align_inputs = pad_sequences(inputs, padding='post',value=0)
        align_answers = pad_sequences(answers, padding='post',value=padding_signal)
        return (align_inputs, align_answers)
    @staticmethod
    def get_seq_vecs(fasta_paths,setting=None):
        seq_converter = None
        discard_invalid_seq = False
        if setting is not None:
            discard_invalid_seq = setting['discard_invalid_sequence']
            seq_converter = SeqConverter(codes=setting['valid_codes'],
                                         with_soft_masked_status=setting['soft_masked'])
        fasta_converter = FastaConverter(seq_converter)
        seq_dict = fasta_converter.to_seq_dict(fasta_paths)
        return fasta_converter.to_vec_dict(seq_dict = seq_dict,discard_invalid_seq = discard_invalid_seq)
    @staticmethod
    def get_ann_vecs(ann_seqs,ann_types):
        warn = ("\n\n!!!\n"
                "\tDNA sequence will be rearranged from 5' to 3'.\n"
                "\tThe plus strand sequence will stay the same,"
                " but the minus strand sequence will be flipped!\n"
                "!!!\n")
        warnings.warn(warn)
        dict_ = {}
        for ann_seq in ann_seqs:
            ann = []
            for type_ in ann_types:
                value = ann_seq.get_ann(type_)
                if ann_seq.strand == 'plus':
                    ann.append(value)
                elif ann_seq.strand == 'minus':
                    ann.append(np.flip(value,0))
                else:
                    raise InvalidStrandType(ann_seq.strand)
            dict_[str(ann_seq.id)] = np.transpose(ann)
        return dict_
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
    def _to_dict(seqs,answer,ann_types):
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
    @classmethod
    def get_data(cls,seq_paths,answer_path,ann_types,setting=None):
        if not isinstance(seq_paths,list):
            seq_paths = [seq_paths]
        seq_paths = [abspath(expanduser(seq_path)) for seq_path in seq_paths]
        seqs = cls.get_seq_vecs(seq_paths,setting)
        ann_seqs = AnnSeqContainer()
        ann_seqs.from_dict(np.load(answer_path).item())
        answer = cls.get_ann_vecs(ann_seqs,ann_types)
        return cls._to_dict(seqs,answer,ann_types)