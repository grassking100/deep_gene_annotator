from abc import ABCMeta, abstractmethod, abstractproperty
from os.path import expanduser
from ..utils.utils import create_folder

class IModelProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self):
        pass
    @abstractproperty
    def model(self):
        pass
    @abstractproperty
    def record(self):
        pass
    def before_process(self,path=None):
        pass
    def after_process(self,path=None):
        pass

class SimpleModel(IModelProcessor):
    def __init__(self,model):
        self._record = {}
        self._model = model
    def process(self):
        pass
    @property
    def model(self):
        return self._model
    @property
    def record(self):
        return self._record


                