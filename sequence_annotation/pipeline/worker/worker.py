"""This submodule provide abstract class about model container"""
from abc import ABCMeta, abstractmethod
import os
from ...utils.exception import ReturnNoneException
from ...utils.helper import create_folder

class Worker(metaclass=ABCMeta):
    """The class hold model and handle with data input, model pedict and model result"""
    def __init__(self, path_root=None):
        self.model = None
        self._result = {}
        self.data = None
        self.is_verbose_visible = True
        self.is_prompt_visible = True
        self._path_root = path_root
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
    @abstractmethod
    def after_work(self):
        """Do something after worker work"""
        pass
    @abstractmethod
    def before_work(self):
        """Do something before worker work"""
        pass
    @abstractmethod
    def _validate(self):
        pass
    def _create_folder(self,folder_names):
        for folder_name in folder_names:
            path = self._path_root+"/"+folder_name
            create_folder(path)