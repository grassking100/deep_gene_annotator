from abc import ABCMeta
from abc import abstractmethod
import pandas as pd
from . import AttrValidator
from . import InvalidAnnotation
from . import AnnSequence,SeqInformation,Sequence
from . import IdNotFoundException,DuplicateIdException
class SeqContainer(metaclass=ABCMeta):
    def __init__(self):
        self._data = {}
        self.note = ""
    def __len__(self):
        return len(self._data)
    def __iter__(self):
        self._index = 0
        self._keys = sorted(list(self._data.keys()))
        return self  
    def __next__(self):
        if self._index >= len(self._data):
            self._index = 0
            self._keys = sorted(list(self._data.keys()))
            raise StopIteration  
        else:
            key = self._keys[self._index]
            self._index += 1  
            return self._data[key]
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
        try:
            iterator = iter(seq_or_seqs)
            for seq in seq_or_seqs:
                self._add(seq)
        except Exception as exp:    
            if hasattr(seq_or_seqs,'to_list'):
                for seq in seq_or_seqs.to_list():
                    self._add(seq)
            else:
                self._add(seq_or_seqs)
    def _add(self, seq):
        self._validate()
        self._validate_seq(seq)
        id_ = seq.id
        if id_ in self._data.keys():
            raise DuplicateIdException(id_)
        self._data[id_] = seq
    def get(self, id_):
        if id_ not in self._data.keys():
            raise IdNotFoundException(id_)  
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
        return self
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
