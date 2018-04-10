from abc import ABCMeta,abstractmethod
import numpy as np
from copy import deepcopy
from tempfile import mkdtemp
import os.path as path
from . import AttrValidator, DictValidator
from . import get_protected_attrs_names
from . import ValueOutOfRange
from . import UninitializedException
from . import NegativeNumberException
from . import InvalidStrandType
from . import InvalidAnnotation
def logical_not(lhs, rhs):
    return np.logical_and(lhs,np.logical_not(rhs))
class Sequence(metaclass=ABCMeta):
    def __init__(self):
        self.note = ""
        self.id = None
        self.source = None
        self.chromosome_id = None
        self._strand = None
    def to_dict(self):
        dictionary = {}
        for attr in self._copied_public_attrs():
            dictionary[attr]=getattr(self,attr)
        for attr in self._copied_protected_attrs():
            dictionary[attr[1:]]=getattr(self,attr)
        return dictionary
    def _validate_dict_keys(self, dict_):
        names = get_protected_attrs_names(dict_)
        validator = DictValidator(dict_,names,[],[])
        validator.validate()
    def from_dict(self, dict_):
        self._validate_dict_keys(dict_)
        for attr in self._copied_public_attrs():
            setattr(self,attr,dict_[attr])
        for attr in self._copied_protected_attrs():
            setattr(self,attr,dict_[attr[1:]])
        return self
    def _copied_public_attrs(self):
        return ['source','chromosome_id','id','note']
    def _copied_protected_attrs(self):
        return ['_strand']
    def _copy(self,source,attrs):
        for attr in attrs:
            setattr(self,attr,deepcopy(getattr(source,attr)))
    @property
    def strand(self):
        return self._strand
    @property
    def valid_strand(self):
        return ['plus','minus']
    @strand.setter
    def strand(self, value):
        if value not in self.valid_strand:
            raise InvalidStrandType(value)
        self._strand = value
    @abstractmethod
    def length(self):
        pass
    def _validate(self):
        attr_validator = AttrValidator(self,False,True,False,None)
        attr_validator.validate()
class SeqInformation(Sequence):
    def __init__(self):
        self._start = None
        self._end = None
        self._extra_index = None
        self.extra_index_name = None
        self.ann_status = None
        self.ann_type = None
        super().__init__()
    def _copied_public_attrs(self):
        return super()._copied_public_attrs()+['ann_type','ann_status','extra_index_name']
    def _copied_protected_attrs(self):
        return super()._copied_protected_attrs()+['_start','_end','_extra_index']
    def _validated_for_length(self):
        attr_validator = AttrValidator(self,False,False,False,['_start','_end'])
        attr_validator.validate()
    @property
    def length(self):
        self._validated_for_length()
        return self._end-self._start+1
    @property
    def start(self):
        return self._start
    @property
    def end(self):
        return self._end
    @start.setter
    def start(self, value):
        if value < 0:
            raise NegativeNumberException("start",value)
        self._start = value
    @end.setter
    def end(self, value):
        if value < 0:
            raise NegativeNumberException("end",value)
        self._end = value
    @property
    def extra_index(self):
        return self._extra_index
    @extra_index.setter
    def extra_index(self, value):
        if value < 0:
            raise NegativeNumberException("extra_index",value)
        self._extra_index = value
class AnnSequence(Sequence):
    def __init__(self):
        self._has_space = False
        self.ANN_TYPES = None
        self._length = None
        self._data = {}
        self._use_memmap = False
        super().__init__()
    def to_dict(self,only_data=False):
        if only_data:
            return self._data
        else:
            return super().to_dict()
    def _copied_public_attrs(self):
        return super()._copied_public_attrs()+['ANN_TYPES','length']
    def _copied_protected_attrs(self):
        return super()._copied_protected_attrs()+['_has_space','_data']
    def from_dict(self, dict_):
        self._validate_dict_keys(dict_)
        for attr in self._copied_public_attrs():
            setattr(self,attr,dict_[attr])
        protected_attrs = list(self._copied_protected_attrs())
        protected_attrs.remove('_data')
        for attr in protected_attrs:
            setattr(self,attr,dict_[attr[1:]])
        if self._has_space:
            if self._use_memmap:
                memmap_id = self.id
            else:
                memmap_id = None
            self.init_space()
            for type_ in self.ANN_TYPES:
                self.set_ann(type_, dict_['data'][type_])
        return self
    @property
    def length(self):
        return self._length
    @length.setter
    def length(self, value):
        if value < 0:
            raise NegativeNumberException("length",value)
        self._length = value
    @property
    def has_space(self):
        return self._has_space
    def init_space(self, memmap_id=None):
        self._use_memmap = (memmap_id is not None)
        self._data = {}
        self._validate()
        self._has_space = True
        for ann_type in self.ANN_TYPES:
            if self._use_memmap:
                filename = path.join(mkdtemp(),str(memmap_id))
                self._data[ann_type] = np.memmap(filename, dtype='float16',
                                                 mode='w+',shape=(self._length))
            else:
                self._data[ann_type] = np.array([0.0]*self._length,dtype='float16')
        return self
    def _validate_input_index(self, start_index, end_index):
        if start_index < 0:
            raise NegativeNumberException('start_index',start_index)
        if end_index < 0 :
            raise NegativeNumberException('end_index',end_index)
        if start_index >= self._length:
            raise ValueOutOfRange('start_index',start_index,range(self._length))
        if end_index >= self._length :
            raise ValueOutOfRange('end_index',end_index,range(self._length))
        if start_index>end_index:
            raise Exception("Start index must not larger than End index")
    def _validate_input_ann_type(self, ann_type):
        if ann_type not in self.ANN_TYPES:
            raise InvalidAnnotation(ann_type)
    def _validate_is_init(self):
        if not self._has_space:
            name = self.__class__.__name__
            raise UninitializedException(name,"Please use method,initSpace")
    def op_and_ann(self, stored_ann_type, masked_ann_type, mask_ann_type,
                   start_index=None, end_index=None):
        self._logical_ann(stored_ann_type, masked_ann_type, mask_ann_type,
                          np.logical_and, start_index, end_index)
        return self
    def op_or_ann(self, stored_ann_type, masked_ann_type, mask_ann_type,
                  start_index=None, end_index=None):
        self._logical_ann(stored_ann_type, masked_ann_type, mask_ann_type,
                          np.logical_or, start_index, end_index)
        return self
    def op_not_ann(self, stored_ann_type, masked_ann_type, mask_ann_type,
                   start_index=None, end_index=None):
        self._logical_ann(stored_ann_type, masked_ann_type, mask_ann_type,
                          logical_not, start_index, end_index)
        return self
    def _logical_ann(self, stored_ann_type, masked_ann_type, mask_ann_type,
                     logical_function, start_index=None, end_index=None):
        masked_ann = self.get_ann(masked_ann_type, start_index, end_index)
        mask_ann = self.get_ann(mask_ann_type, start_index, end_index)
        mask = logical_function(masked_ann,mask_ann)
        masked_ann[np.logical_not(mask)] = 0
        self.set_ann(stored_ann_type, masked_ann, start_index, end_index)
    def add_ann(self, ann_type, value, start_index=None, end_index=None):
        ann = self.get_ann(ann_type, start_index, end_index)
        self.set_ann(ann_type, ann + value, start_index, end_index)
        return self
    def set_ann(self,ann_type, value, start_index=None, end_index=None):
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self._length -1
        self._validate_is_init()
        self._validate_input_ann_type(ann_type)
        self._validate_input_index(start_index, end_index)
        self._data[ann_type][start_index : end_index+1] = value
        if self._use_memmap:      
            self._data[ann_type].flush()
        return self
    def get_ann(self,ann_type, start_index=None, end_index=None):
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self._length - 1
        self._validate_is_init()
        self._validate_input_ann_type(ann_type)
        self._validate_input_index(start_index, end_index)
        return self._data[ann_type][start_index : end_index+1].copy()
    def get_normalized(self, frontground_types, background_type):
        self._validate_is_init()
        ann_seq = AnnSequence().from_dict(self.to_dict())
        values = []
        frontground_seq = self._get_frontground(ann_seq,frontground_types)
        background_seq = self._get_background(ann_seq,frontground_types)
        for type_ in frontground_types:
            values.append(ann_seq.get_ann(type_))
        sum_values = np.array(values).sum(0)
        for type_ in frontground_types:
            numerator = self.get_ann(type_)
            denominator = sum_values
            with np.errstate(divide='ignore', invalid='ignore'):
                result = numerator / denominator
                result[denominator == 0] = 0
                result[np.logical_not(frontground_seq)] = 0
                ann_seq.set_ann(type_,result)
        ann_seq.set_ann(background_type,background_seq)
        return ann_seq
    def get_one_hot(self, frontground_types,background_type):
        ann_seq = self.get_normalized(frontground_types,background_type)
        values = []
        one_hot_value = {}
        for type_ in ann_seq.ANN_TYPES:
            values.append(ann_seq.get_ann(type_))
            one_hot_value[type_] =[0]*ann_seq.length
        one_hot_indice = np.argmax(values,0)
        for index,one_hot_index in enumerate(one_hot_indice):
            one_hot_type = ann_seq.ANN_TYPES[one_hot_index]
            one_hot_value[one_hot_type][index]=1
        for type_ in frontground_types:
            ann_seq.set_ann(type_,one_hot_value[type_])
        return ann_seq
    def _get_background(self,ann_seq,frontground_types):
        return  np.logical_not(self._get_frontground(ann_seq,frontground_types))
    def _get_frontground(self,ann_seq,frontground_types):
        frontground_seq = np.array([0]*ann_seq.length)
        for type_ in frontground_types:
            frontground_seq = np.logical_or(frontground_seq,ann_seq.get_ann(type_))
        return frontground_seq