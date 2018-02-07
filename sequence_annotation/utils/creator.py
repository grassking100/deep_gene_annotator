"""This submodule provide abstract class about creator"""
from abc import ABCMeta, abstractmethod, abstractproperty
class Creator(metaclass=ABCMeta):
    """The class provide api for create result"""
    def __init__(self):
        self._result = None
    @abstractmethod
    def _validate(self):
        pass
    @abstractmethod
    def create(self):
        """Method to build object"""
        pass
    @abstractproperty
    def result(self):
        pass
