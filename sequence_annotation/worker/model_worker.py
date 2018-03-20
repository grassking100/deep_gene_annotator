"""This submodule provide abstract class about model container"""
from abc import ABCMeta, abstractmethod
import os
from . import ReturnNoneException
class ModelWorker(metaclass=ABCMeta):
    """The class hold model and handle with data input, model pedict and model result"""
    def __init__(self):
        self.model = None
        self.result = {}
        self.data = {}
    def init_worker(self, path_root, is_verbose_visible=True, is_prompt_visible=True):
        self._path_root = path_root
        self._is_verbose_visible = is_verbose_visible
        self._is_prompt_visible = is_prompt_visible
    def clean_result(self):
        """clean result"""
        self.result = {}
    @abstractmethod
    def work(self):
        """Work"""
        pass
    @abstractmethod
    def after_work(self):
        """Do something after worker work"""
        pass
    def before_work(self):
        """Do something before worker work"""
        pass
    def _create_folder(self,folder_names):
        for folder_name in folder_names:
            path = self._path_root+"/"+folder_name
            if not os.path.exists(path):
                os.makedirs(path)
