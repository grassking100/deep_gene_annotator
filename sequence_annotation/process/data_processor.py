import warnings
import random
import numpy as np
from ..genome_handler.seq_container import AnnSeqContainer
from ..genome_handler import ann_genome_processor
from ..utils.seq_converter import SeqConverter
from ..utils.exception import LengthNotEqualException,DimensionNotSatisfy
from ..utils.utils import get_subdict

class AnnSeqProcessor:
    def __init__(self,data,seq_converter=None,
                 discard_invalid_seq=False,validation_split=None,ann_types=None):
        self._data = data
        if ann_types is None:
            if 'training' in data.keys():
                self._ann_types = data['training']['answers'].ANN_TYPES
            else:
                self._ann_types = data['testing']['answers'].ANN_TYPES
        else:
            self._ann_types = ann_types
        self._validation_split = validation_split or 0
        if seq_converter is None:
            self._seq_converter = SeqConverter()
        else:
            self._seq_converter = seq_converter
        self._discard_invalid_seq = discard_invalid_seq

    def _validate(self,data):
        for id_,input_,answer in zip(data['ids'],data['inputs'],data['answers']):
            seq_length = np.shape(input_)[0]
            ann_length = np.shape(answer)[0]
            if ann_length != seq_length:
                raise LengthNotEqualException(ann_length, seq_length, id_)

    def _split(self):
        if self._validation_split > 0 and not 'validation' in self._data.keys():
            shuffled_keys = list(self._data['training']['inputs'].keys())
            random.shuffle(shuffled_keys)
            val_length = int(len(shuffled_keys)*self._validation_split)
            train_keys = shuffled_keys[val_length:]
            val_keys = shuffled_keys[:val_length]
            train_seqs = {}
            val_seqs = {}
            data = self._data['training']
            for type_,item in data.items():
                if isinstance(item,AnnSeqContainer):
                    train_seqs[type_] = item.get_seqs(train_keys)
                    val_seqs[type_] = item.get_seqs(val_keys)
                else:
                    train_seqs[type_] = get_subdict(train_keys,item)
                    val_seqs[type_] = get_subdict(val_keys,item)
            self._data['training'] = train_seqs
            self._data['validation'] = val_seqs

    def _to_dict(self,item):
        seqs = self._seq_converter.seqs2dict_vec(item['inputs'],self._discard_invalid_seq)
        ann_seq_dict = ann_genome_processor.genome2dict_vec(item['answers'],self._ann_types)
        data = {'inputs':[],'answers':[],'lengths':[],'ids':[],'seqs':[]}
        for name in seqs.keys():
            seq = seqs[name]
            answer = ann_seq_dict[name]
            data['ids'].append(name)
            data['seqs'].append(item['inputs'][name])
            data['inputs'].append(seq)
            data['answers'].append(answer)
            data['lengths'].append(len(seq))
        return data

    def process(self):
        self._split()
        warning = "{} data have {} sequences, it left {} sequences after filtering"
        for purpose in self._data.keys():
            item = self._data[purpose]
            origin_num = len(item['inputs'])
            item = self._to_dict(item)
            new_num = len(item['inputs'])
            if origin_num != new_num:
                warnings.warn(warning.format(purpose,origin_num,new_num),UserWarning)
            self._data[purpose] = item
            self._validate(self._data[purpose])
        return self._data
