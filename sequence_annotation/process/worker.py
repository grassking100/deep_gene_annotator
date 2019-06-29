"""This submodule provide abstract class about model container"""
from abc import ABCMeta, abstractmethod

class Worker(metaclass=ABCMeta):
    """The class hold model and handle with data input, model pedict and model result"""
    def __init__(self):
        self.model = None
        self.data = None
        self.result = {}
        self.is_verbose_visible = True
        self.is_prompt_visible = True
        self._settings = {}

    @abstractmethod
    def work(self):
        """Work"""
        pass

    def after_work(self,**kwargs):
        """Do something after worker work"""
        pass

    def before_work(self,**kwargs):
        """Do something before worker work"""
        pass

    def _validate(self):
        """Validate required data"""
        if self.model is None:
            raise Exception("Model must be set")
        if self.data is None:
            raise Exception("Data must be set")
            