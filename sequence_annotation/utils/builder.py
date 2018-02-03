"""This submodule provide abstract class about builder"""
from abc import ABCMeta, abstractmethod
class Builder(metaclass=ABCMeta):
    """The class provide api for building object"""
    @abstractmethod
    def _validate(self):
        pass
    @abstractmethod
    def build(self):
        """Method to build object"""
        pass
