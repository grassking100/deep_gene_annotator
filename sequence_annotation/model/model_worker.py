"""This submodule provide abstract class about model container"""
from abc import ABCMeta, abstractmethod
class ModelWorker(metaclass=ABCMeta):
    """The class hold model and handle with data input, model pedict and model result"""
    def __init__(self):
        self._model = None
        self._result = {}
        self._valid_data_keys = []
        self._data = {}
    @property
    def result(self):
        """Get result"""
        if self._result is None:
            raise Exception("There is not result yet")
        return self._result
    @result.setter
    def result(self, result):
        """Set result"""
        self._result = result
    def clean_result(self):
        """clean result"""
        self.result = {}
    def _validate_keys(self, key):
        if key in self._valid_data_keys:
            raise Exception("Invalid key:"+key) 
    def load_data(self, data, kind):
        """Check for its validity and load data to container"""
        self._validate_keys(kind)
        self._data[kind] = data
    @property
    def model(self):
        """Get model"""
        return self._model
    @model.setter
    def model(self, model):
        """Set model"""
        self._model = model
    @abstractmethod
    def _validate(self):
        """Validate"""
        pass
