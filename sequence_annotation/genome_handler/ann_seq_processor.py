from abc import ABCMeta,abstractmethod
import numpy as np
from . import AnnSequence
from . import UninitializedException
class AnnSeqProcessor(metaclass=ABCMeta):
    def is_dirty(self,seq,dirty_types):
        for dirty_type in dirty_types:
            if np.sum(seq.get_ann(dirty_type)) > 0:
                return True
        return False
    def combine_status(self,ann_seq,status_dict):
        extra_types = list(status_dict.keys())
        combined_seq = AnnSequence()
        combined_seq.from_dict(ann_seq.to_dict(without_data=True))
        combined_seq.processed_status = None
        combined_seq.ANN_TYPES = ann_seq.ANN_TYPES + extra_types
        combined_seq.clean_space()
        combined_seq.init_space()
        for type_ in ann_seq.ANN_TYPES:
            combined_seq.set_ann(type_,ann_seq.get_ann(type_))
        for key,value in status_dict.items():
            combined_seq.set_ann(key,value)
        return combined_seq
    def get_certain_status(self,ann_seq,focus_types=None):
        focus_types = focus_types or ann_seq.ANN_TYPES
        ann = ann_seq.to_dict(only_data=True)
        data = [ann[type_] for type_ in focus_types]
        certain_status = np.ceil(np.array(data)).sum(axis=0)==1
        return np.array(certain_status,dtype='bool')
    def _get_focus_types(self, seq_ann_types,frontground_types, background_type):
        frontground_types_set = set(frontground_types)
        background_type_set = set([background_type])
        focus_type_set = list(frontground_types_set.union(background_type_set))
        return focus_type_set
    def _get_unfocus_types(self, seq_ann_types,frontground_types, background_type):
        focus_types = self._get_focus_types(seq_ann_types,
                                                frontground_types,
                                                background_type)
        other_type = list(set(seq_ann_types).difference(set(focus_types)))
        return other_type
    def get_normalized(self, seq, frontground_types, background_type):
        if seq.processed_status=='normalized':
            return seq
        else:
            if len(set(frontground_types).intersection(set(background_type)))!=0:
                raise Exception("Frontground_types and background_type has duplicated type")
            other_types = self._get_unfocus_types(seq.ANN_TYPES,frontground_types, background_type)
            norm_seq = AnnSequence().from_dict(seq.to_dict())
            norm_seq.processed_status='normalized'
            values = []
            frontground_seq = self._get_frontground(seq,frontground_types)
            background_seq = self._get_background(seq,frontground_types)
            for type_ in frontground_types:
                values.append(norm_seq.get_ann(type_))
            sum_values = np.array(values).sum(0)
            for type_ in frontground_types:
                numerator = seq.get_ann(type_)
                denominator = sum_values
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = numerator / denominator
                    result[denominator == 0] = 0
                    result[np.logical_not(frontground_seq)] = 0
                    norm_seq.set_ann(type_,result)
            for type_ in other_types:
                norm_seq.set_ann(type_,seq.get_ann(type_))
            norm_seq.set_ann(background_type,background_seq)
            return norm_seq
    def get_one_hot(self, seq, frontground_types, background_type):
        if seq.processed_status=='one_hot':
            return seq
        else:
            if len(set(frontground_types).intersection(set(background_type)))!=0:
                raise Exception("Frontground_types and background_type has duplicated type")
            one_hot_seq = self.get_normalized(seq,frontground_types,background_type)
            one_hot_seq.processed_status='one_hot'
            other_types = self._get_unfocus_types(seq.ANN_TYPES,frontground_types, background_type)
            values = []
            one_hot_value = {}
            for type_ in seq.ANN_TYPES:
                values.append(one_hot_seq.get_ann(type_))
                one_hot_value[type_] =[0]*one_hot_seq.length
            one_hot_indice = np.argmax(values,0)
            for index,one_hot_index in enumerate(one_hot_indice):
                one_hot_type = seq.ANN_TYPES[one_hot_index]
                one_hot_value[one_hot_type][index]=1
            for type_ in frontground_types:
                one_hot_seq.set_ann(type_,one_hot_value[type_])
            for type_ in other_types:
                one_hot_seq.set_ann(type_,seq.get_ann(type_))
            return one_hot_seq
    def _get_background(self,ann_seq,frontground_types):
        return  np.logical_not(self._get_frontground(ann_seq,frontground_types))
    def _get_frontground(self,ann_seq,frontground_types):
        frontground_seq = np.array([0]*ann_seq.length)
        for type_ in frontground_types:
            frontground_seq = np.logical_or(frontground_seq,ann_seq.get_ann(type_))
        return frontground_seq