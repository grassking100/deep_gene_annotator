from abc import ABCMeta,abstractmethod
from . import MissingExpectDictKey, InvalidValueInDict, AttrIsNoneException, LengthNotEqualException, DictKeyNotExistException
from . import get_protected_attrs_names
import numpy as np
class IVaildable(metaclass=ABCMeta):
    @abstractmethod
    def validate(self):
        pass
class AttrValidator(IVaildable):
    """A class provides API for chcking if passed object's attribute is None or not"""
    def __init__(self, object_,
                 is_private_validated,
                 is_protected_validated,
                 is_public_validated,
                 validate_attr_names=None):
        self._object= object_
        self._is_private_validated = is_private_validated
        self._is_public_validated = is_public_validated
        self._is_protected_validated = is_protected_validated
        self._validate_attr_names = validate_attr_names or []
    def validate(self):
        attrs = self._validate_attr_names.copy()
        if self._is_private_validated:
            attrs += self._private_attr_names()
        if self._is_public_validated:
            attrs += self._public_attr_names()
        if self._is_protected_validated:
            attrs += self._protected_attr_names()
        self._validateAttr(attrs)
    def _validateAttr(self,attr_names):
        """Validate if attributes is not None"""
        for attr in attr_names:
            if getattr(self._object, attr) is None:
                name = self._object.__class__.__name__
                raise AttrIsNoneException(attr, name)
    def _private_attr_names(self):
        class_name = self._object.__class__.__name__
        attrs = [attr for attr in dir(self._object) if attr.startswith('_'+class_name+'__')]
        return attrs
    def _protected_attr_names(self):
        return get_protected_attrs_names(self._object)
    def _public_attr_names(self):
        attrs = [attr for attr in dir(self._object) if not attr.startswith('_')]
        return attrs
class DictValidator(IVaildable):
    """A class provides API for chcking if dictionay is Valid or not"""
    def __init__(self,dictionay,keys_must_included,keys_of_validated_value,invalid_values):
        self._dict = dictionay
        self._keys_must_included = keys_must_included
        for key in keys_of_validated_value:
            self._validate_key_exist(key)
        self._keys_of_validated_value = keys_of_validated_value
        self._invalid_values = invalid_values
    def _validate_key_exist(self,key):
        if key not in list(self._dict.keys()):
            raise DictKeyNotExistException(key)
    def _validate_keys(self):
        for key in self._keys_must_included:
            if key not in self._dict.keys():
                raise MissingExpectDictKey(key)
    def _validate_values(self):
        for key, value in self._dict.items():
            if key in self._keys_of_validated_value:
                for invalid_value in self._invalid_values:
                    if np.any(value==invalid_value):
                        raise InvalidValueInDict(key, invalid_value)
    def validate(self):
        self._validate_keys()
        self._validate_values()
class DataValidator():
    def same_shape(self, input_sequencse, output_sequences):
        """
            Validate if length of input and output sequence is same.
            If size is not same it will raise a Exception.
        """
        input_sequencse_length = input_sequencse.shape[1]
        output_sequencse_length = output_sequences.shape[1]
        if input_sequencse_length != output_sequencse_length:
            raise LengthNotEqualException(str(input_sequencse_length),
                                          str(output_sequencse_length))
