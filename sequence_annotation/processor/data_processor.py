from abc import ABCMeta, abstractmethod,abstractproperty
import numpy as np
import random
from ..genome_handler.seq_container import AnnSeqContainer
from ..data_handler.seq_converter import SeqConverter
from ..genome_handler import ann_seq_processor
from ..utils.exception import LengthNotEqualException,DimensionNotSatisfy
from ..utils.utils import create_folder,padding

class IDataProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self):
        pass
    @abstractproperty
    def record(self):
        pass
    @abstractproperty
    def data(self):
        pass
    @abstractmethod
    def before_process(self,path=None):
        pass
    @abstractmethod
    def after_process(self,path=None):
        pass

"""class DataLoader(IDataProcessor):
    def __init__(self,path):
        self._data = None
        self._path = path
        self._record = {}
    @property
    def record(self):
        return self._record
    @property
    def data(self):
        return self._data
    def before_process(self,path=None):
        pass
    def after_process(self,path=None):
        pass
    def process(self):
        pass
"""

class SimpleData(IDataProcessor):
    def __init__(self,data):
        self._data = data
        self._record = {}
    @property
    def record(self):
        return self._record
    @property
    def data(self):
        return self._data
    def before_process(self,path=None):
        pass
    def after_process(self,path=None):
        pass
    def process(self):
        pass

class AnnSeqData(SimpleData):
    def __init__(self,data,padding_value=None,seq_converter=None,
                 discard_invalid_seq=False,validation_split=0,do_validate=True):
        super().__init__(data['data'])
        self._record['padding_value'] = padding_value
        self._record['discard_invalid_seq'] = discard_invalid_seq
        self._record['validation_split'] = validation_split
        self._record['do_validate'] = do_validate
        self._validation_split = validation_split
        self._seq_converter = seq_converter or SeqConverter()
        self._padding_value = padding_value
        self._discard_invalid_seq = discard_invalid_seq
        self._ANN_TYPES = data['ANN_TYPES']
        self._do_validate = do_validate
    def before_process(self,path=None):
        if path is not None:
            json_path = create_folder(path) + "/setting/data.json"
            with open(json_path,'w') as fp:
                json.dump(self._record,fp)
    def _padding(self):
        padded = {}
        for data_kind, data in self._data.items():
            inputs = data['inputs']
            answers = data['answers']
            inputs, answers = padding(inputs,answers,self._padding_value)
            padded[data_kind] = {"inputs":inputs,"answers":answers}
        self._data =  padded
    def _to_dict(self):
        for data_kind,data in self._data.items(): 
            seqs = data['inputs']
            answers = data['answers']
            seqs = self._seq_converter.encode_seqs(seqs,self._discard_invalid_seq)
            answers = ann_seq_processor.get_ann_vecs(answers,self._ANN_TYPES)
            data_pairs = {'inputs':[],'answers':[]}
            for name in seqs.keys():
                seq = seqs[name]
                answer = answers[name]
                ann_length = np.shape(seq)[0]
                seq_length = np.shape(answer)[0]
                if ann_length != seq_length:
                    raise LengthNotEqualException(ann_length, seq_length)
                data_pairs['inputs'].append(seq)
                data_pairs['answers'].append(answer)
            self._data[data_kind] = data_pairs
    def _validate(self):
        for data_kind,data in self._data.items():
            input_shape = np.shape(data['inputs'])
            answer_shape = np.shape(data['answers'])
            if len(input_shape)!=3:
                raise DimensionNotSatisfy(input_shape,3)
            if len(answer_shape)!=3:
                raise DimensionNotSatisfy(answer_shape,3)
    def _split(self):
        if self._validation_split > 0 and not 'validation' in self._data.keys():
            ANN_TYPES = self._data['training']['answers'].ANN_TYPES
            shuffled_keys = list(self._data['training']['inputs'].keys())
            random.shuffle(shuffled_keys)
            val_length = int(len(shuffled_keys)*self._validation_split)
            train_keys = shuffled_keys[val_length:]
            val_keys = shuffled_keys[:val_length]
            train_ann_seqs = AnnSeqContainer()
            train_ann_seqs.ANN_TYPES = ANN_TYPES
            val_ann_seqs = AnnSeqContainer()
            val_ann_seqs.ANN_TYPES = ANN_TYPES
            train_seqs = {'inputs':{},'answers':train_ann_seqs}
            val_seqs = {'inputs':{},'answers':val_ann_seqs}
            for key in train_keys:
                train_seqs['inputs'][key] = self._data['training']['inputs'][key] 
            for key in val_keys:
                val_seqs['inputs'][key] = self._data['training']['inputs'][key]
            for key in train_keys:
                train_seqs['answers'].add(self._data['training']['answers'][key])
            for key in val_keys:
                val_seqs['answers'].add(self._data['training']['answers'][key])
            self._data['training'] = train_seqs
            self._data['validation'] = val_seqs
    def process(self):
        self._split()
        self._to_dict()
        if self._padding_value is not None:
            self._padding()
        if self._do_validate:
            self._validate()
                          
class DataContainerFactory:
    def create(self,type_):
        if type_ == 'simple':
            return DataContainer
        elif type_ == 'ann_seq':
            return AnnSeqLoader
        else:
            raise Exception(type_+' has not be supported yet.')
