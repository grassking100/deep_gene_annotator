"""This submodule provide abstract class about Extractor"""
from abc import ABCMeta, abstractmethod, abstractproperty
class Extractor(metaclass=ABCMeta):
    """The class provide api for extract result"""
    def __init__(self):
        self._result = None
    @abstractmethod
    def _validate(self):
        pass
    @abstractmethod
    def extract(self):
        """Method to build object"""
        pass
    @abstractproperty
    def result(self):
        pass
