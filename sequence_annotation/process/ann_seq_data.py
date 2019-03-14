import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
from ..genome_handler.sequence import AnnSequence
from ..genome_handler.seq_container import AnnSeqContainer
from ..genome_handler.utils import get_subseqs
from ..data_handler.seq_converter import SeqConverter
from ..genome_handler import ann_genome_processor,ann_seq_processor
from ..utils.exception import LengthNotEqualException,DimensionNotSatisfy
from ..utils.utils import create_folder,get_subdict
from .data_processor import SimpleData


class AnnSeqData(SimpleData):
    def __init__(self,data,padding=None,seq_converter=None,answer_by_index=False,
                 discard_invalid_seq=False,validation_split=0):
        super().__init__(data)
        if 'training' in data.keys():
            self._ann_types = data['training']['answers'].ANN_TYPES
        else:
            self._ann_types = data['testing']['answers'].ANN_TYPES
        self._record['padding'] = padding
        self._record['seq_converter'] = seq_converter
        self._record['answer_by_index'] = answer_by_index
        self._record['discard_invalid_seq'] = discard_invalid_seq
        self._record['validation_split'] = validation_split
        self._validation_split = validation_split
        self._seq_converter = seq_converter or SeqConverter()
        self._padding = padding or {}
        self._discard_invalid_seq = discard_invalid_seq
        self._answer_by_index = answer_by_index

    def before_process(self,path=None):
        if path is not None:
            json_path = create_folder(path) + "/setting/data.json"
            with open(json_path,'w') as fp:
                json.dump(self._record,fp)

    def _validate(self,data_dict):
        for input_,answer in zip(data_dict['inputs'],data_dict['answers']):
            ann_length = np.shape(input_)[0]
            seq_length = np.shape(answer)[0]
            if ann_length != seq_length:
                raise LengthNotEqualException(ann_length, seq_length)
        if 'inputs' in self._padding.keys():
            input_shape = np.shape(data_dict['inputs'])
            if len(input_shape) != 3:
                raise DimensionNotSatisfy(input_shape,3)
        if 'answer' in self._padding.keys():
            answer_shape = np.shape(data_dict['answers'])
            if self._answer_by_index:
                if len(answer_shape) != 3:
                    raise DimensionNotSatisfy(answer_shape,3)
            else:
                if len(answer_shape) != 2:
                    raise DimensionNotSatisfy(answer_shape,2)

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
                    train_seqs[type_] = get_subseqs(train_keys,item)
                    val_seqs[type_] = get_subseqs(val_keys,item)
                else:
                    train_seqs[type_] = get_subdict(train_keys,item)
                    val_seqs[type_] = get_subdict(val_keys,item)
            self._data['training'] = train_seqs
            self._data['validation'] = val_seqs

    def _handle_extra(self,data):
        new_data = {}
        preserved_key = ['inputs','answers']
        new_data['extra'] = {}
        for kind,item in data.items():
            if kind in preserved_key:
                new_data[kind] = item
            else:
                new_data['extra'][kind] = item
        return new_data

    def _pad(self,data):
        padded = {}
        for kind in data.keys():
            if kind in self._padding.keys():
                temp = pad_sequences(data[kind],padding='post',value=self._padding[kind])
            else:
                temp = data[kind]
            padded[kind] = temp
        return padded

    def _to_dict(self,item):
        seqs = self._seq_converter.seqs2dict_vec(item['inputs'],self._discard_invalid_seq)
        ann_seq_dict_ = ann_genome_processor.genome2dict_vec(item['answers'],self._ann_types)
        ann_seq_dict = {}
        if self._answer_by_index:
            for id_,ann in ann_seq_dict_.items():
                ann_seq_dict[id_] = np.argmax(ann,axis=-1)
        else:
            ann_seq_dict = ann_seq_dict_
        item['inputs'] = seqs
        item['answers'] = ann_seq_dict
        dict_list = {'inputs':[],'answers':[],'lengths':[],'ids':[]}
        other_key = [key for key in item.keys() if key not in ['inputs','answers']]
        for name in seqs.keys():
            seq = seqs[name]
            answer = ann_seq_dict[name]
            dict_list['ids'].append(name)
            dict_list['inputs'].append(seq)
            dict_list['answers'].append(answer)
            dict_list['lengths'].append(len(seq))
        for data_kind in other_key:
            dict_list[data_kind] = []
            temp = item[data_kind]
            if isinstance(temp,AnnSeqContainer):
                temp = ann_genome_processor.genome2dict_vec(temp)
            for name in seqs.keys():
                dict_list[data_kind].append(temp[name])
        return dict_list
            
    def process(self):
        self._split()
        for purpose in self._data.keys():
            self._data[purpose] = self._to_dict(self._data[purpose])
            if self._padding is not None:
                self._data[purpose] = self._pad(self._data[purpose])
            self._data[purpose] = self._handle_extra(self._data[purpose])
            self._validate(self._data[purpose])