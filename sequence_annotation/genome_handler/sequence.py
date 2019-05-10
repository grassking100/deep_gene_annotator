from abc import ABCMeta,abstractproperty
import numpy as np
from copy import deepcopy
from tempfile import mkdtemp
import os.path as path
from ..utils.validator import AttrValidator, DictValidator
from ..utils.utils import get_protected_attrs_names,logical_not
from ..utils.exception import UninitializedException,NegativeNumberException,InvalidStrandType
from ..utils.exception import InvalidAnnotation,ChangeConstValException,ValueOutOfRange

class Sequence(metaclass=ABCMeta):
    def __init__(self):
        self.note = ""
        self.id = None
        self.source = None
        self.chromosome_id = None
        self._strand = None
    def __len__(self):
        return self.length
    def to_dict(self):
        dictionary = {}
        for attr in self._copied_public_attrs():
            dictionary[attr]=deepcopy(getattr(self,attr))
        for attr in self._copied_protected_attrs():
            dictionary[attr[1:]]=deepcopy(getattr(self,attr))
        return dictionary
    def _validate_dict_keys(self, dict_):
        names = get_protected_attrs_names(dict_)
        validator = DictValidator(dict_,names,[],[])
        validator.validate()
    def from_dict(self, dict_):
        self._validate_dict_keys(dict_)
        for attr in self._copied_public_attrs():
            setattr(self,attr,deepcopy(dict_[attr]))
        for attr in self._copied_protected_attrs():
            setattr(self,attr,deepcopy(dict_[attr[1:]]))
        return self
    def _copied_public_attrs(self):
        return ['source','chromosome_id','id','note']
    def _copied_protected_attrs(self):
        return ['_strand']
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
    @abstractproperty
    def length(self):
        pass
    def _checked_attr(self):
        return ['id','_strand']
    def _validate(self):
        attr_validator = AttrValidator(self,False,False,False,self._checked_attr())
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
        self._ANN_TYPES = None
        self._length = None
        self._data = {}
        self._absolute_index = None
        self.processed_status = None
        super().__init__()
    def _checked_attr(self):
        return super()._checked_attr()+['_ANN_TYPES','_length']
    @property
    def absolute_index(self):
        return self._absolute_index
    @absolute_index.setter
    def absolute_index(self, value):
        if value < 0:
            raise NegativeNumberException("absolute_index",value)
        self._absolute_index = value
    @property
    def ANN_TYPES(self):
        return self._ANN_TYPES
    @ANN_TYPES.setter
    def ANN_TYPES(self,value):
        if self._ANN_TYPES is None or not self._has_space:
            if len(set(value))!=len(value):
                raise Exception('Input types has duplicated')
            self._ANN_TYPES = list(value)
            self._ANN_TYPES.sort()
        else:
            raise ChangeConstValException('ANN_TYPES')
    def _copied_public_attrs(self):
        return super()._copied_public_attrs()+['ANN_TYPES','length','processed_status']
    def _copied_protected_attrs(self):
        return super()._copied_protected_attrs()+['_has_space','_absolute_index']
    def to_dict(self,only_data=False,without_data=False):
        if only_data:
            return self._data
        else:
            dict_ = super().to_dict()
            if not without_data:
                protected_attrs = list(self._copied_protected_attrs())
                protected_attrs.append('_data')
                for attr in protected_attrs:
                    dict_[attr[1:]]=deepcopy(getattr(self,attr))
            else:
                dict_['data'] = {}
                dict_['has_space']=False
            return dict_
    def from_dict(self, dict_):
        self._validate_dict_keys(dict_)
        super().from_dict(dict_)
        if self._has_space:
            self.init_space()
            for type_ in self.ANN_TYPES:
                value = deepcopy(dict_['data'][type_])
                self.set_ann(type_, value)
        return self
    @property
    def length(self):
        return self._length
    @length.setter
    def length(self, value):
        if value < 0:
            raise NegativeNumberException("length",value)
        if not self._has_space:
            self._length = value
        else:
            raise ChangeConstValException('length')
    def clean_space(self):
        self._has_space = False
        self._data = {}
        self.processed_status = None
    @property
    def has_space(self):
        return self._has_space
    def init_space(self, dtype='float32'):
        self._data = {}
        self._validate()
        self.processed_status = None
        self._has_space = True
        for ann_type in self.ANN_TYPES:
            self._data[ann_type] = np.array([0.0]*self._length,dtype=dtype)
        return self
    def _validate_input_index(self, start_index, end_index):
        if start_index < 0:
            raise NegativeNumberException('start_index',start_index)
        if end_index < 0 :
            raise NegativeNumberException('end_index',end_index)
        if start_index >= self._length:
            raise ValueOutOfRange('start_index',start_index,list([0,self._length-1]))
        if end_index >= self._length :
            raise ValueOutOfRange('end_index',end_index,list([0,self._length-1]))
        if start_index>end_index:
            raise Exception("Start index must not larger than End index")
    def _validate_input_ann_type(self, ann_type):
        if ann_type not in self.ANN_TYPES:
            raise InvalidAnnotation(ann_type)
    def _validate_is_init(self):
        if not self._has_space:
            name = self.__class__.__name__
            raise UninitializedException(name,"Please use method,init_space")
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
        return_ann = logical_function(masked_ann,mask_ann)
        self.set_ann(stored_ann_type, return_ann, start_index, end_index)
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

    def get_subseq(self, start_index=None, end_index=None):
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self._length - 1
        sub_seq = self.__class__().from_dict(self.to_dict())
        sub_seq.clean_space()
        sub_seq.length = end_index - start_index + 1
        sub_seq.init_space()
        for type_ in self.ANN_TYPES:
            sub_seq.set_ann(type_,self.get_ann(type_,start_index,end_index))
        return sub_seq
