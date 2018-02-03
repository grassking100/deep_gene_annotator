from abc import ABCMeta
from abc import abstractmethod
from . import AttrValidator
from . import InvalidAnnotation
import pandas as pd
class SeqContainer(metaclass=ABCMeta):
    def __init__(self):
        self._dict = {}
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
        unsorted_seqs = list(self._dict.values())
        sorted_seqs = sorted(unsorted_seqs, key=lambda seq: seq.id)
        return sorted_seqs
    @abstractmethod
    def _validate_seq(self,seq):
        pass
    @abstractmethod
    def _validate(self):
        pass
    def to_data_frame(self):
        df = pd.DataFrame.from_dict(self._dict,'index')
        return df
    def add(self, seq):
        self._validate()
        self._validate_seq(seq)
        id_ = seq.id
        if id_ in self._dict.keys():
            raise Exception("ID," + str(id_) + ", is duplicated")
        self._dict[id_] = seq
    def get(self, id_):
        if id_ not in self._dict.keys():
            raise Exception("There is no sequence about " + id_)    
        return self._dict[id_]
class SeqInfoContainer(SeqContainer):
    def _validate(self):
        pass
    def _validate_seq(self,seq):
        pass
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
class AnnGenomeManipulator():
    def to_normalized(self, ann_seq_genome):
        genome = AnnSeqContainer()
        genome.ANN_TYPES = ann_seq_genome.ANN_TYPES
        genome.note="normalized"
        for seq in ann_seq_genome.data:
            genome.add(seq.get_normalized())
        return genome 
    def to_one_hot(self, ann_seq_genome, background_ann_type):
        genome = AnnSeqContainer()
        genome.ANN_TYPES = ann_seq_genome.ANN_TYPES
        genome.note="one-hot"
        for seq in genome.data:
            genome.add(seq.get_one_hot(background_ann_type))
        return genome