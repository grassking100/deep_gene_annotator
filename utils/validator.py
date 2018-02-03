from abc import ABCMeta,abstractmethod
from . import MissingExpectDictKey, InvalidValueInDict, AttrIsNoneException, LengthNotEqualException, DictKeyNotExistException
class IVaildable(metaclass=ABCMeta):
    @abstractmethod
    def validate(self):
        pass
class AttrValidator(IVaildable):
    """A class provides API for chcking if passed object's attribute is None or not"""
    def __init__(self, object_):
        self._object= object_
        self._is_private_validated = False
        self._is_public_validated = False
        self._is_protected_validated = False
        self._validate_attr_names = []
    @property
    def validate_attr_names(self):
        return self._validate_attr_names
    @property
    def is_private_validated(self):
        return self._is_private_validated
    @property
    def is_public_validated(self):
        return self._is_public_validated
    @property
    def is_protected_validated(self):
        return self._is_protected_validated
    @is_private_validated.setter
    def is_private_validated(self, value):
        self._is_private_validated = value
    @is_public_validated.setter
    def is_public_validated(self, value):
        self._is_public_validated = value
    @is_protected_validated.setter
    def is_protected_validated(self, value):
        self._is_protected_validated = value
    @validate_attr_names.setter
    def validate_attr_names(self, value):
        self._validate_attr_names = value
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
        class_name = self._object.__class__.__name__
        attrs = [attr for attr in dir(self._object) if attr.startswith('_') 
                and not attr.endswith('__') and not attr.startswith('_'+class_name+'__')]
        return attrs
    def _public_attr_names(self):
        attrs = [attr for attr in dir(self._object) if not attr.startswith('_')]
        return attrs
class DictValidator(IVaildable):
    """A class provides API for chcking if dictionay is Valid or not"""
    def __init__(self,dictionay):
        self._dict = dictionay
        self._keys_must_included = []
        self._keys_of_validated_value = []
        self._invalid_values = []
    @property
    def keys_of_validated_value(self):
        return self._keys_of_validated_value
    @keys_of_validated_value.setter
    def keys_of_validated_value(self, key):
        self._validate_key_exist(key)
        self._keys_of_validated_value = key
    @property
    def keys_must_included(self):
        return self._keys_must_included
    @keys_must_included.setter
    def key_must_included(self, value):
        self._keys_must_included = value
    @property
    def invalid_values(self):
        return self._invalid_values
    def _validate_key_exist(self,key):
        if key not in self._dict.keys():
            raise DictKeyNotExistException(key)
    @invalid_values.setter
    def invalid_values(self, value):
        self._invalid_values = value
    def _validate_keys(self):
        for key in self.keys_must_included:
            if key not in self._dict.keys():
                raise MissingExpectDictKey(key)
    def _validate_values(self):
        for key, value in self._dict.items():
            if key in self.keys_of_validated_value:
                for invalid_value in self.invalid_values:
                    if value == invalid_value:
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
