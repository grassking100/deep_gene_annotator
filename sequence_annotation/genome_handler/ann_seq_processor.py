from abc import ABCMeta,abstractmethod
import numpy as np
from . import ProcessedStatusNotSatisfied
from . import AnnSequence
from . import UninitializedException
class NotOneHotException(Exception):
    def __init__(self,seq_id):
        msg = "Sequence,"+str(seq_id)+",is not one hot encoded"
        super().__init__(msg)

class AnnSeqProcessor(metaclass=ABCMeta):
    def is_dirty(self,seq,dirty_types):
        for dirty_type in dirty_types:
            if np.sum(seq.get_ann(dirty_type)) > 0:
                return True
        return False
    def get_certain_status(self,seq,focus_types=None):
        if not seq.processed_status == 'normalized':
            raise ProcessedStatusNotSatisfied(seq.processed_status,'normalized')
        focus_types = focus_types or seq.ANN_TYPES
        ann = seq.to_dict(only_data=True)
        data = [ann[type_] for type_ in focus_types]
        certain_status = np.ceil(np.array(data)).sum(axis=0)==1
        return np.array(certain_status,dtype='bool')       
    def _get_unfocus_types(self, ann_types,focus_types):
        other_type = list(set(ann_types).difference(set(focus_types)))
        return other_type
    def get_normalized(self, seq, focus_types=None):
        if seq.processed_status=='normalized':
            return seq
        else:
            focus_types = focus_types or seq.ANN_TYPES
            if not self.is_full_annotated(seq,focus_types):
                raise Exception("Sequence is not fully annotated")
            focus_types = focus_types or seq.ANN_TYPES
            other_types = self._get_unfocus_types(seq.ANN_TYPES,focus_types)
            norm_seq = AnnSequence().from_dict(seq.to_dict())
            norm_seq.processed_status='normalized'
            values = []
            frontground_seq = self.get_frontground(seq,focus_types)
            for type_ in focus_types:
                values.append(norm_seq.get_ann(type_))
            sum_values = np.array(values).sum(0)
            for type_ in focus_types:
                numerator = seq.get_ann(type_)
                denominator = sum_values
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = numerator / denominator
                    result[denominator == 0] = 0
                    result[np.logical_not(frontground_seq)] = 0
                    norm_seq.set_ann(type_,result)
            for type_ in other_types:
                norm_seq.set_ann(type_,seq.get_ann(type_))
            return norm_seq    
    def is_full_annotated(self, seq, focus_types=None):
        focus_types = focus_types or seq.ANN_TYPES
        values = []
        for type_ in focus_types:
            values.append(seq.get_ann(type_))
        status = not np.any(np.array(values).sum(0)==0) 
        return status
    def is_value_sum_to_length(self, seq, focus_types=None):
        focus_types = focus_types or seq.ANN_TYPES
        values = [0]*seq.length
        for type_ in focus_types:
            values += seq.get_ann(type_)
        return sum(values)==seq.length
    def is_one_hot(self, seq, focus_types=None):
        focus_types = focus_types or seq.ANN_TYPES
        values = []
        for type_ in focus_types:
            values.append(seq.get_ann(type_))
        is_0_1 = np.all(np.isin(np.array(values), [0,1]))
        is_full_annotated = self.is_full_annotated(seq, focus_types)
        is_sum_to_length = self.is_value_sum_to_length(seq, focus_types)
        return is_0_1 and is_full_annotated and is_sum_to_length
    def _get_one_hot_by_max(self, seq, focus_types=None):
        focus_types = focus_types or seq.ANN_TYPES
        one_hot_seq = self.get_normalized(seq,focus_types)
        one_hot_seq.processed_status = 'one_hot'
        other_types = self._get_unfocus_types(seq.ANN_TYPES,focus_types)
        values = []
        one_hot_value = {}
        for type_ in focus_types:
            values.append(one_hot_seq.get_ann(type_))
            one_hot_value[type_] = [0]*one_hot_seq.length
        one_hot_indice = np.argmax(values,0)
        for index,one_hot_index in enumerate(one_hot_indice):
            one_hot_type = focus_types[one_hot_index]
            one_hot_value[one_hot_type][index] = 1
        for type_ in focus_types:
            one_hot_seq.set_ann(type_,one_hot_value[type_])
        for type_ in other_types:
            one_hot_seq.set_ann(type_,seq.get_ann(type_))
        return one_hot_seq
    def _get_one_hot_by_order(self, seq, focus_types=None):
        focus_types = focus_types or seq.ANN_TYPES
        one_hot_seq = self.get_normalized(seq,focus_types)
        one_hot_seq.processed_status='one_hot'
        other_types = self._get_unfocus_types(seq.ANN_TYPES,focus_types)
        temp = AnnSequence().from_dict(one_hot_seq.to_dict())
        temp.clean_space()
        temp.ANN_TYPES = ['focused','focusing','temp']
        temp.init_space()
        for type_ in focus_types:
            temp.set_ann('focusing',one_hot_seq.get_ann(type_))
            temp.op_not_ann('temp','focusing','focused')
            temp.op_or_ann('focused','focusing','focused')
            one_hot_seq.set_ann(type_,temp.get_ann('temp'))
        for type_ in other_types:
            one_hot_seq.set_ann(type_,seq.get_ann(type_))
        return one_hot_seq
    def get_one_hot(self, seq, focus_types=None, method='max'):
        if seq.processed_status=='one_hot':
            return seq
        else:
            focus_types = focus_types or seq.ANN_TYPES
            if not self.is_full_annotated(seq,focus_types):
                raise Exception("Sequence is not fully annotated")
            if method == 'max':
                return self._get_one_hot_by_max(seq,focus_types)
            elif method == 'order':
                return self._get_one_hot_by_order(seq,focus_types)
            else:
                raise Exception("Method ,"+str(method)+", is not supported")
    def get_background(self,seq,frontground_types=None):
        frontground_types = frontground_types or seq.ANN_TYPES
        return  np.logical_not(self.get_frontground(seq,frontground_types))
    def get_seq_with_added_type(self,ann_seq,status_dict):
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
    def get_frontground(self,seq,frontground_types=None):
        frontground_types = frontground_types or seq.ANN_TYPES
        frontground_seq = np.array([0]*seq.length)
        for type_ in frontground_types:
            frontground_seq = np.logical_or(frontground_seq,seq.get_ann(type_))
        return frontground_seq