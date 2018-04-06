"""This submodule provide abstract class about model container"""
from abc import ABCMeta, abstractmethod
import os
from . import ReturnNoneException
class Worker(metaclass=ABCMeta):
    """The class hold model and handle with data input, model pedict and model result"""
    def __init__(self, path_root):
        self._model = None
        self._result = {}
        self._data = None
        self.is_verbose_visible = True
        self.is_prompt_visible = True
        self._path_root = path_root
    @property
    def result(self):
        return self._result
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
    def _create_folder(self,folder_names):
        for folder_name in folder_names:
            path = self._path_root+"/"+folder_name
            if not os.path.exists(path):
                os.makedirs(path)
