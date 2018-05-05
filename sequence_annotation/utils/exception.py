class ChangeConstValException(Exception):
    def __init__(self, name = None):
        type_=""
        if name is not None:
            type_ = ","+str(name)+","
        msg = ("Try to change constant value"+str(type_))
        super().__init__(msg)
class InvalidAnnotation(Exception):
    def __init__(self, ann_type = None):
        type_=""
        if ann_type is not None:
            type_ = ","+str(ann_type)+","
        msg = ("Annotation type"+str(type_)+" "
               "is not expected")
        super().__init__(msg)
class ValueOutOfRange(Exception):
    def __init__(self, value_name,value,range_):
        msg =("Value,"+value_name+", must be in"
              " "+str(range_)+",your value is "+str(value))
        super().__init__(msg)
class NegativeNumberException(ValueError):
    def __init__(self, value_name, value):
        msg =(value_name+","+str(value)+", must not be negative")
        super().__init__(msg)
class NotPositiveException(ValueError):
    def __init__(self, value_name, value):
        msg =(value_name+","+str(value)+", must be positive")
        super().__init__(msg)

class DictKeyNotExistException(Exception):
    def __init__(self, key):
        msg = ("Key,"+str(key)+", doesn't exist in "
               "dictionary")
        super().__init__(msg)
class MissingExpectDictKey(Exception):
    def __init__(self, missing_key):
        msg = ("Passed dictionary needs data about "
               ""+str(missing_key)+" to complete the quest")
        super().__init__(msg)
class InvalidValueInDict(Exception):
    def __init__(self, key, invalid_value):
        msg = ("Passed dictionary has invilad value"
               ","+str(invalid_value) + " , in " + str(key))
        super().__init__(msg)
class AttrIsNoneException(Exception):
    def __init__(self, attr_name, class_name=None):
        if class_name is None:
            class_name = "Class"
        msg = (str(class_name) + "'s attribute"
               "," + str(attr_name) + ", should "
               "not be None")
        super().__init__(msg)
class InvalidStrandType(Exception):
    def __init__(self, strand_type= None):
        type_=""
        if strand_type is not None:
            type_ = ","+ str(strand_type)+","
        msg = ("Strand type"+str(type_)+" "
               "is not expected")
        super().__init__(msg)
class LengthNotEqualException(Exception):
    def __init__(self, first_length, second_length):
        msg = ("Two data must have same length"
               ", size of first data is"
               " "+str(first_length)+","
               "and size of second data is"
               " "+str(second_length))
        super().__init__(msg)
class ReturnNoneException(Exception):
    def __init__(self,attr_name,solution=None):
        msg = "Try to return "+attr_name+" with None value"
        if solution is not None:
            msg +=(","+str(solution))
        super().__init__(msg)
class UninitializedException(Exception):
    def __init__(self,class_name,solution):
        msg = class_name + " has not be initialized"
        msg +=(","+str(solution))
        super().__init__(msg)

class CodeException(Exception):
    """Raise when input code is not in in defined space"""
    def __init__(self,invalid_code,valid_codes=None):
        self._invalid_code = invalid_code
        self._valid_codes = valid_codes
        mess = str(invalid_code)+' is not in defined space'
        if self._valid_codes is not None:
            mess+=(",valid codes are "+str(self._valid_codes))
        super().__init__(mess)
    @property
    def invalid_code(self):
        return self._invalid_code
    @property
    def valid_codes(self):
        return self._valid_codes
class SeqException(Exception):
    """Raise when input sequences has at least a code is not in in defined space"""
    def __init__(self,invalid_code,valid_codes=None):
        mess = "Seqeunce has a invalid code,"+str(invalid_code)
        if valid_codes is not None:
            mess+=(" ,valid codes are "+str(valid_codes))
        super().__init__(mess)