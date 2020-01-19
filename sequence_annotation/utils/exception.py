class IdNotFoundException(Exception):
    def __init__(self,missing_id):
        msg = "ID," + str(missing_id) + ", is not found"
        super().__init__(msg)

class DuplicateIdException(Exception):
    def __init__(self,duplicated_id):
        msg = "ID," + str(duplicated_id) + ", is duplicated"
        super().__init__(msg)

class ChangeConstValException(Exception):
    def __init__(self, name = None):
        type_=""
        if name is not None:
            type_ = ","+str(name)
        msg = ("Try to change constant value"+str(type_))
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

class AttrIsNoneException(Exception):
    def __init__(self, attr_name, class_name=None):
        if class_name is None:
            class_name = "Class"
        msg = (str(class_name) + "'s attribute"
               "," + str(attr_name) + ", should "
               "not be None")
        super().__init__(msg)

class LengthNotEqualException(Exception):
    def __init__(self, first_length, second_length,id_=None):
        msg = "Two data must have same length, size of first data is {}, and size of second data is {}"
        msg = msg.format(first_length,second_length)
        if id_ is not None:
             msg += ', this error occurs in {}'.format(id_) 
        super().__init__(msg)

class UninitializedException(Exception):
    def __init__(self,class_name,solution):
        msg = class_name + " has not be initialized"
        msg +=(","+str(solution))
        super().__init__(msg)

class CodeException(Exception):
    """Raising when input code is not in in defined space"""
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
    """Raising when input sequences has at least a code is not in in defined space"""
    def __init__(self,invalid_code,valid_codes=None):
        mess = "Seqeunce has a invalid code,"+str(invalid_code)
        if valid_codes is not None:
            mess+=(" ,valid codes are "+str(valid_codes))
        super().__init__(mess)