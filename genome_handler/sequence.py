from abc import ABCMeta
from abc import abstractmethod, abstractproperty
import numpy as np
from . import AttrValidator
from . import ValueOutOfRange, NegativeNumberException, InvalidStrandType, InvalidAnnotation, UninitializedException
def bitwise_not(lhs, rhs):
    return np.bitwise_and(lhs,np.logical_not(rhs))
class Sequence(metaclass=ABCMeta):
    def __init__(self):
        self._id = None
        self._source = None
        self._chromosome_id = None
        self._strand = None
        self._note = ""
    def to_dict(self):
        dictionary = {}
        dictionary['id'] = self._id
        dictionary['source'] = self._source
        dictionary['chromosome_id'] = self._chromosome_id
        dictionary['strand'] = self._strand
        dictionary['note'] = self._note
        return dictionary
    def copy(self):
        return self._copy(self._copied_attrs())
    def _copied_attrs(self):
        return ['_source','_chromosome_id','_strand','_note','_id']
    def _copy(self,attrs):
        seq = self.__class__()
        for attr in attrs:
            setattr(seq,attr,getattr(self,attr))
        return seq
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
    @strand.setter
    def strand(self, value):
        if value not in ['plus','minus']:
            raise InvalidStrandType(value)
        self._strand = value
    @abstractproperty
    def length(self):
        pass
    @abstractmethod
    def _validate(self):
        pass
class SeqInformation(Sequence):
    def __init__(self):
        super().__init__()
        self._start = None
        self._end = None
        self._extra_index = None
        self._extra_index_name = None
        self._ann_type = None
        self._ann_status = None
    def _copied_attrs(self):
        return super()._copied_attrs()+['_start','_end',
                                        '_ann_type','_ann_status',
                                        '_extra_index',
                                        '_extra_index_name']
    def to_dict(self):
        dictionary = super().to_dict()
        dictionary['start'] = self._start
        dictionary['end'] = self._end
        dictionary['extra_index'] = self._extra_index
        dictionary['extra_index_name'] = self._extra_index_name
        dictionary['ann_type'] = self._ann_type
        dictionary['ann_status'] = self._ann_status
        return dictionary
    def _validate(self):
        pass
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
        attr_validator = AttrValidator(self)
        attr_validator.validated_attr = ['_start','_end']
        attr_validator.validate()
    @property
    def length(self):
        self._validated_for_length()
        return self._end-self._start
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
    def __init__(self,):
        super().__init__()
        self._ANN_TYPES = None
        self._data = None
        self._length = None
        self._has_space = False
    def to_dict(self):
        dictionary = super().to_dict()
        dictionary['ANN_TYPES'] = self._ANN_TYPES
        dictionary['data'] = self._data
        dictionary['length'] = self._length
        dictionary['has_space'] = self._has_space
        return dictionary
    def _copied_attrs(self):
        return super()._copied_attrs()+['_ANN_TYPES','_length']
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
        return self.has_space
    def _validate(self):
        attr_validator = AttrValidator(self)
        attr_validator.is_protected_validated = True
        attr_validator.validate()
    def initSpace(self):
        self._data = {}
        self._validate()
        self._has_space = True
        for ann_type in self._ANN_TYPES:
            self._data[ann_type] = np.array([0]*self._length)
    def _validate_input_index(self, start_index, end_index):
        if start_index < 0:
            raise NegativeNumberException('start_index',start_index)
        if end_index < 0 :
            raise NegativeNumberException('end_index',end_index)
        if start_index >= self._length:
            raise ValueOutOfRange('start_index',start_index,range(self._length))
        if end_index >= self._length :
            raise ValueOutOfRange('end_index',end_index,range(self._length))
    def _validate_input_ann_type(self, ann_type):
        if ann_type not in self._ANN_TYPES:
            raise InvalidAnnotation(ann_type)
    def _validate_is_init(self):
        if not self._has_space:
            name = self.__class__.__name__
            raise UninitializedException(name,"Please use method,initSpace")
    def op_and_ann(self, lhs_ann_type, rhs_1_ann_type, rhs_2_ann_type,
                   start_index=None, end_index=None):
        self._bitwise_ann(lhs_ann_type, rhs_1_ann_type, rhs_2_ann_type,
                          np.bitwise_and, start_index, end_index)
    def op_or_ann(self, lhs_ann_type, rhs_1_ann_type, rhs_2_ann_type,
                  start_index=None, end_index=None):
        self._bitwise_ann(lhs_ann_type, rhs_1_ann_type, rhs_2_ann_type,
                          np.bitwise_or, start_index, end_index)
    def op_not_ann(self, lhs_ann_type, rhs_1_ann_type, rhs_2_ann_type,
                   start_index=None, end_index=None):
        self._bitwise_ann(lhs_ann_type, rhs_1_ann_type, rhs_2_ann_type,
                          bitwise_not, start_index, end_index)    
    def _bitwise_ann(self, lhs_ann_type, rhs_1_ann_type, rhs_2_ann_type,
                     bitwise_function, start_index=None, end_index=None):
        rhs_1_ann = self.get_ann(rhs_1_ann_type, start_index, end_index)
        rhs_2_ann = self.get_ann(rhs_2_ann_type, start_index, end_index)
        result = bitwise_function(rhs_1_ann,rhs_2_ann)
        self.set_ann(lhs_ann_type, result, start_index, end_index)
    def add_ann(self, ann_type, value, start_index=None, end_index=None):
        ann = self.get_ann(ann_type, start_index, end_index)
        self.set_ann(ann_type, ann + value, start_index, end_index)
    def set_ann(self,ann_type, value, start_index=None, end_index=None):
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self._length -1
        self._validate_is_init()
        self._validate_input_ann_type(ann_type)
        self._validate_input_index(start_index, end_index)
        self._data[ann_type][start_index : end_index+1]=value
    def get_ann(self,ann_type, start_index=None, end_index=None):
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self._length -1
        self._validate_is_init()
        self._validate_input_ann_type(ann_type)
        self._validate_input_index(start_index, end_index)
        return self._data[ann_type][start_index : end_index+1]
    def get_normalized(self):
        self._validate_is_init()
        temp_ann_seq = self.copy()
        temp_ann_seq.initSpace()
        values = np.array(list(self.data.values()))
        nomralized_values = values / values.sum(0)
        np.nan_to_num(nomralized_values)
        normalized_data = dict((ann_type,value) for ann_type,value in zip(self.ANN_TYPES,nomralized_values))
        for ann_type in self._ANN_TYPES:
            temp_ann_seq.set_ann(ann_type,normalized_data[ann_type])
        return temp_ann_seq
    def _handle_background(self,ann_seq,background_type):
        frontground_seq = np.array([0]*ann_seq.length)
        frontground_types = [type_ for type_ in ann_seq.ANN_TYPES if type_ is not background_type]
        for type_ in frontground_types:
            frontground_seq = np.bitwise_or(frontground_seq,ann_seq.get_ann(type_))
        background_seq = np.invert(frontground_seq)
        ann_seq.set_seq(background_type,background_seq)
    def get_one_hot(self,background_type):
        ann_seq =self.get_normalized()
        self._handle_background(ann_seq,background_type)
        values = np.array(list(ann_seq.data.values()))
        one_hot_value = np.array([0],self._length*len(self._ANN_TYPES))
        one_hot_value.shape = (self._ANN_TYPES,self._length)
        one_hot_value[values.argmax(0),np.arange(self._length)]=1.0
        one_hot_data =dict((ann_type,value) for ann_type,value in zip(self._ANN_TYPES,one_hot_value))
        for ann_type in self._ANN_TYPES:
            ann_seq.set_ann(ann_type,one_hot_data[ann_type])
        return ann_seq 
