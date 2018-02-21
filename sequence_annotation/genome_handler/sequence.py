from abc import ABCMeta
import numpy as np
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
    def __init__(self,object_=None):
        self._note = ""
        self._id = None
        self._source = None
        self._chromosome_id = None
        self._strand = None
        if object_ is not None:
            self._copy(object_,self._copied_attrs())
    def to_dict(self):
        dictionary = {}
        for attr in self._copied_attrs():
            dictionary[attr]=getattr(self,"_"+attr)
        return dictionary
    def _validate_dict_keys(self, dict_):
        names = get_protected_attrs_names(dict_)
        validator = DictValidator(dict_,names,[],[])
        validator.validate()
    def from_dict(self, dict_):
        self._validate_dict_keys(dict_)
        for attr in self._copied_attrs():
            setattr(self,"_"+attr,dict_[attr])
    def _copied_attrs(self):
        return ['source','chromosome_id','strand','id','note']
    def _copy(self,source,attrs):
        for attr in attrs:
            setattr(self,"_"+attr,getattr(source,attr))
    @property
    def id(self):
        return self._id
    @id.setter
    def id(self, value):
        self._id = value
    @property
    def note(self):
        return self._note
    @note.setter
    def note(self, value):
        self._note = value
    @property
    def source(self):
        return self._source
    @property
    def chromosome_id(self):
        return self._chromosome_id
    @property
    def strand(self):
        return self._strand
    @source.setter
    def source(self, value):
        self._source = value
    @chromosome_id.setter
    def chromosome_id(self, value):
        self._chromosome_id = value
    @property
    def valid_strand(self):
        return ['plus','minus']
    @strand.setter
    def strand(self, value):
        if value not in self.valid_strand:
            raise InvalidStrandType(value)
        self._strand = value
    def length(self):
        return None
    def _validate(self):
        attr_validator = AttrValidator(self,False,True,False,None)
        attr_validator.validate()
class SeqInformation(Sequence):
    def __init__(self,object_=None):
        self._start = None
        self._end = None
        self._extra_index = None
        self._extra_index_name = None
        self._ann_type = None
        self._ann_status = None
        super().__init__(object_)
    def _copied_attrs(self):
        return super()._copied_attrs()+['start','end',
                                        'ann_type','ann_status',
                                        'extra_index',
                                        'extra_index_name']
    @property
    def ann_type(self):
        return self._ann_type
    @property
    def ann_status(self):
        return self._ann_status
    @ann_type.setter
    def ann_type(self, value):
        self._ann_type = value
    @ann_status.setter
    def ann_status(self, value):
        self._ann_status = value
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
    @property
    def extra_index_name(self):
        return self._extra_index
    @extra_index_name.setter
    def extra_index_name(self, value):
        self._extra_index_name = value
class AnnSequence(Sequence):
    def __init__(self,object_=None):
        self._data = None
        self._has_space = False
        self._ANN_TYPES = None
        self._length = None
        self._data = {}
        super().__init__(object_)
        """if object_ is not None:
            
            if self._has_space:
                for type_ in self._ANN_TYPES:
                    self._data[type_] = getattr(object_,'_data')[type_]"""
    def _copied_attrs(self):
        return super()._copied_attrs()+['ANN_TYPES','length','has_space','data']
    @property
    def data(self):
        return self._data
    @property
    def ANN_TYPES(self):
        return self._ANN_TYPES
    @property
    def length(self):
        return self._length
    @ANN_TYPES.setter
    def ANN_TYPES(self, value):
        self._ANN_TYPES = value
    @length.setter
    def length(self, value):
        if value < 0:
            raise NegativeNumberException("length",value)
        self._length = value
    @property
    def has_space(self):
        return self._has_space
    def initSpace(self):
        self._data = {}
        self._validate()
        self._has_space = True
        for ann_type in self._ANN_TYPES:
            self._data[ann_type] = np.array([0.0]*self._length)
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
        if ann_type not in self._ANN_TYPES:
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
        self._data[ann_type][start_index : end_index+1]=value
        return self
    def get_ann(self,ann_type, start_index=None, end_index=None):
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self._length -1
        self._validate_is_init()
        self._validate_input_ann_type(ann_type)
        self._validate_input_index(start_index, end_index)
        return self._data[ann_type][start_index : end_index+1].copy()
    def get_normalized(self, frontground_types, background_type):
        self._validate_is_init()
        ann_seq = AnnSequence(self)
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