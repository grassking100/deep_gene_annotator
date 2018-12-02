"""This submodule provide abstract class about model container"""
from abc import ABCMeta, abstractmethod
import os
from ..utils.exception import ReturnNoneException
from ..utils.utils import create_folder

class IWorker(metaclass=ABCMeta):
    """The class hold model and handle with data input, model pedict and model result"""
    @abstractmethod
    def __init__(self, path_root=None):
        pass
    @property
    def result(self):
        return self._result
    @result.setter
    def result(self,value):
        self._result = value
    @abstractmethod
    def work(self):
        """Work"""
        pass
    def after_work(self):
        """Do something after worker work"""
        pass
    def before_work(self):
        """Do something before worker work"""
        pass
    @abstractmethod
    def _validate(self):
        pass
    
class Worker(IWorker):
    """The class hold model and handle with data input, model pedict and model result"""
    def __init__(self, path_root=None):
        self.model = None
        self._result = {}
        self.data = None
        self.is_verbose_visible = True
        self.is_prompt_visible = True
        self._path_root = path_root
        self.wrapper = None
    @abstractmethod
    def work(self):
        """Work"""
        pass
    def after_work(self):
        """Do something after worker work"""
        pass
    def before_work(self):
        """Do something before worker work"""
        pass
    def _validate(self):
        """Validate required data"""
        if self.model is None:
            raise Exception("Model must be passed to worker")
        if self.data is None:
            raise Exception("Data must be passed to worker")
        if self.wrapper is None:
            raise Exception("Wrapper must be passed to worker")
            