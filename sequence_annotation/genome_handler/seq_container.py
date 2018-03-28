from abc import ABCMeta
from abc import abstractmethod
import pandas as pd
from . import AttrValidator
from . import InvalidAnnotation
from . import AnnSequence,SeqInformation,Sequence
class SeqContainer(metaclass=ABCMeta):
    def __init__(self):
        self._data = {}
        self.note = ""
    @property
    def data(self):
        """Return sequences in order based on their id"""
        unsorted_seqs = list(self._data.values())
        sorted_seqs = sorted(unsorted_seqs, key=lambda seq: seq.id)
        return sorted_seqs
    def to_list(self):
        return self.data
    @abstractmethod
    def _validate_seq(self,seq):
        pass
    @abstractmethod
    def _validate(self):
        pass
    def add(self,seq_or_seqs):
        if type(seq_or_seqs) == list:
            for seq in seq_or_seqs:
                self._add(seq)
        elif hasattr(seq_or_seqs,'to_list'):
            for seq in seq_or_seqs.to_list():
                self._add(seq)
        else:
            self._add(seq_or_seqs)
    def _add(self, seq):
        self._validate()
        self._validate_seq(seq)
        id_ = seq.id
        if id_ in self._data.keys():
            raise Exception("ID," + str(id_) + ", is duplicated")
        self._data[id_] = seq
    def get(self, id_):
        if id_ not in self._data.keys():
            raise Exception("There is no sequence about " + id_)  
        return self._data[id_]
    def to_data_frame(self):
        data = []
        for item in self.data:
            data.append(item.to_dict())
        return pd.DataFrame().from_dict(data)
    def to_dict(self):
        dict_ = {"data":[],"note":self.note}
        for item in self.data:
            dict_['data'] += [item.to_dict()]
        return dict_
    def _create_sequence(self):
        return Sequence()
    def from_dict(self, dict_):
        for data in dict_['data']:
            seq = self._create_sequence()
            seq.from_dict(data)
            self.add(seq)
        self.note = dict_['note']
class SeqInfoContainer(SeqContainer):
    def _validate(self):
        pass
    def _validate_seq(self,seq):
        pass
    def _create_sequence(self):
        return SeqInformation()
    def to_gtf(self):
        df = self.to_data_frame()
        selected_df = df[['id','source','strand']].copy()
        selected_df['seqname'] = df['chromosome_id']
        selected_df['start'] = df['start'] + 1
        selected_df['end'] = df['end'] + 1
        selected_df['feature'] = df['id']
        selected_df['score'] = '.'
        selected_df['frame'] = '.'
        selected_df['attribute'] = "seed_status="+df['ann_type']+"_"+df['ann_status']
        gtf_order = ['seqname','source','feature',
                     'start','end','score',
                     'strand','frame','attribute']
        selected_df['strand']=selected_df['strand'].str.replace("plus", '+')
        selected_df['strand']=selected_df['strand'].str.replace("minus", '-')
        return selected_df[gtf_order]
class AnnSeqContainer(SeqContainer):
    def __init__(self):
        super().__init__()
        self.ANN_TYPES = None
    def _validate_seq(self, seq):
        diffs = set(seq.ANN_TYPES).symmetric_difference(self.ANN_TYPES)
        if len(diffs) > 0:
            raise InvalidAnnotation(str(diffs))
    def _validate(self):
        validator = AttrValidator(self,False,True,False,None)
        validator.validate()
    def _create_sequence(self):
        return AnnSequence()
    def from_dict(self,dict_):
        self.ANN_TYPES = dict_["type"]
        super().from_dict(dict_)
    def to_dict(self):
        dict_ = super().to_dict()
        dict_["type"] = self.ANN_TYPES
        return dict_
    def data_to_dict(self):
        dict_ = {}
        for item in self.data:
            dict_[item.id]=item.data
        return dict_