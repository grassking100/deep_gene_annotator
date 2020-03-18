from abc import ABCMeta
from abc import abstractmethod
import pandas as pd
from ..utils.exception import IdNotFoundException,DuplicateIdException,AttrIsNoneException,ChangeConstValException
from ..utils.utils import GFF_COLUMNS
from .exception import InvalidAnnotation,InvalidStrandType
from .sequence import AnnSequence, SeqInformation, PLUS, MINUS

class EmptyContainerException(Exception):
    def __init__(self):
        super().__init__("Container is empty")

class SeqContainer(metaclass=ABCMeta):
    def __init__(self):
        self._data = {}
        self.note = ""
        self._keys = None
        self._index = None

    @property
    def ids(self):
        return list(self._data.keys())

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        self._index = 0
        self._keys = sorted(self.ids)
        return self

    def __next__(self):
        if self._index >= len(self._data) or self.is_empty():
            self._index = 0
            self._keys = sorted(self.ids)
            raise StopIteration
        else:
            key = self._keys[self._index]
            self._index += 1
            return self._data[key]

    def is_empty(self):
        return len(self) == 0

    def clean(self):
        self._data = {}

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

    def add(self,seqs):
        if isinstance(seqs,self.__class__) or isinstance(seqs,list):
            for seq in seqs:
                self._add(seq)
        else:
            self._add(seqs)

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

    @abstractmethod
    def _create_sequence(self):
        pass

    def from_dict(self, dict_):
        self.clean()
        for data in dict_['data']:
            seq = self._create_sequence()
            seq.from_dict(data)
            self.add(seq)
        self.note = dict_['note']
        return self

    def __setitem__(self, id_, seq):
        if seq.id is None:
            seq.id = id_
            self._add(seq)
        else:
            if seq.id == id_:
                self._add(seq)
            else:
                raise Exception('Sequence\'s id is not same as inputed key')

    def __getitem__(self, id_):
        return self.get(id_)

    def copy(self,with_seq=True):
        new_container = self.__class__()
        new_container.note = self.note
        if with_seq:
            for seq in self._data.values():
                new_container._data[seq.id] = seq.copy()
        return new_container
    
    def get_seqs(self,ids):
        seqs = self.copy(with_seq=False)
        return seqs.add([seqs[id_] for id_ in ids])

class SeqInfoContainer(SeqContainer):
    def _validate(self):
        pass

    def _validate_seq(self,seq):
        pass

    def _create_sequence(self):
        return SeqInformation()

    def to_gff(self):
        if self.is_empty():
            raise EmptyContainerException()
        df = self.to_data_frame()
        selected_df = df[['id','source','strand']].copy()
        selected_df['source']=selected_df['source'].str.replace("", '.')
        selected_df['chr'] = df['chromosome_id']
        selected_df['start'] = df['start'] + 1
        selected_df['end'] = df['end'] + 1
        selected_df['feature'] = df['ann_type']
        selected_df['score'] = '.'
        selected_df['frame'] = '.'
        selected_df['attribute'] = "ID="+df['id'].astype(str)+";Parent="+df['parent'].astype(str)+";Status="+df['ann_status'].astype(str)
        selected_df['strand']=selected_df['strand'].str.replace(PLUS, '+')
        selected_df['strand']=selected_df['strand'].str.replace(MINUS, '-')
        return selected_df[GFF_COLUMNS]
    
    def from_gff(self,gff):
        if len(gff) == 0:
            raise Exception("GFF is empty")
        gff.rename(columns={"chr": "chromosome_id", "feature": "ann_type"})
        gff = gff.to_dict('record')
        strands = set(gff['strand'])
        if strands != set(['+','-']):
            raise InvalidStrandType(strands)
        for item in gff:
            item['start'] -= 1
            item['end'] -= 1
            item['strand'] = itme['strand'].replace("+",PLUS).replace("-",MINUS)
            for attribute in item['attribute'].split(";"):
                if attribute.startswith("ID="):
                    item['attribute'] = attribute[3:]
                    break
            seq = self._create_sequence()
            seq.from_dict(item)
            self.add(seq)
        return self

class AnnSeqContainer(SeqContainer):
    def __init__(self,ann_types=None):
        super().__init__()
        self._ANN_TYPES = None
        if ann_types is not None:
            self.ANN_TYPES = ann_types

    @property
    def ANN_TYPES(self):
        return self._ANN_TYPES

    @ANN_TYPES.setter
    def ANN_TYPES(self,value):
        if self._ANN_TYPES is None:
            if len(set(value))!=len(value):
                raise Exception('Input types has duplicated data')
            self._ANN_TYPES = list(value)
            self._ANN_TYPES.sort()
        else:
            raise ChangeConstValException('ANN_TYPES')

    def _validate_seq(self, seq):
        diffs = set(seq.ANN_TYPES).symmetric_difference(self.ANN_TYPES)
        if len(diffs) != 0:
            raise InvalidAnnotation(str(diffs))

    def _validate(self):
        if self.ANN_TYPES is None:
            raise AttrIsNoneException(self.ANN_TYPES, 'ANN_TYPES')

    def _create_sequence(self):
        return AnnSequence()

    def from_dict(self,dict_):
        self.ANN_TYPES = dict_["type"]
        return super().from_dict(dict_)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["type"] = self.ANN_TYPES
        return dict_

    def add(self,seqs):
        if self.ANN_TYPES is not None:
            super().add(seqs)
        else:
            raise Exception("AnnSeqContainer's ANN_TYPES must not be None")

    def copy(self,with_seq=True):
        new_container = super().copy(with_seq)
        new_container.ANN_TYPES = self._ANN_TYPES
        return new_container
