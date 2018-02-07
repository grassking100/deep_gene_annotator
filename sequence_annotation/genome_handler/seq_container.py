from abc import ABCMeta
from abc import abstractmethod
from . import AttrValidator
from . import InvalidAnnotation
import pandas as pd
class SeqContainer(metaclass=ABCMeta):
    def __init__(self):
        self._data = {}
        self._note = ""
    @property
    def note(self):
        return self._note
    @note.setter
    def note(self, value):
        self._note = value
    @property
    def data(self):
        """Return sequences in order based on their id"""
        unsorted_seqs = list(self._data.values())
        sorted_seqs = sorted(unsorted_seqs, key=lambda seq: seq.id)
        return sorted_seqs
    @abstractmethod
    def _validate_seq(self,seq):
        pass
    @abstractmethod
    def _validate(self):
        pass
    def to_data_frame(self):
        df = pd.DataFrame.from_dict(self._data,'index')
        return df
    def add(self,seq_seqs):
        if type(seq_seqs) == list:
            for seq in seq_seqs:
                self._add(seq)
        else:
            self._add(seq_seqs)
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
class SeqInfoContainer(SeqContainer):
    def _validate(self):
        pass
    def _validate_seq(self,seq):
        pass
    def to_gtf(self):
        df = self.to_data_frame()
        selected_df = df[['id','source','strand']].copy()
        selected_df['seqname'] = df['chromosome_id']
        selected_df['start'] = df['start'] + 1
        selected_df['end'] = df['end'] + 1
        selected_df['feature'] = df['id']
        selected_df['score'] = '.'
        selected_df['frame'] = '.'
        selected_df['attribute'] = "seed_id="+df['ann_type']+"_"+df['ann_status']
        gtf_order = ['seqname','source','feature',
                     'start','end','score',
                     'strand','frame','attribute']
        selected_df['strand']=selected_df['strand'].str.replace("plus", '+')
        selected_df['strand']=selected_df['strand'].str.replace("minus", '-')
        return selected_df[gtf_order]
class AnnSeqContainer(SeqContainer):
    def __init__(self):
        super().__init__()
        self._ANN_TYPES = None
    @property
    def ANN_TYPES(self):
        return self._ANN_TYPES
    @ANN_TYPES.setter
    def ANN_TYPES(self, values):
        self._ANN_TYPES = values
    def _validate_seq(self, seq):
        for ann_type in seq.ANN_TYPES:
            if ann_type not in self._ANN_TYPES:
                raise InvalidAnnotation(ann_type)
    def _validate(self):
        validator = AttrValidator(self)
        validator.is_protected_validated = True
        validator.validate()
    def to_dict(self):
        dict_ = {}
        for item in self.data:
            dict_[item.id] = item.data
        return dict_