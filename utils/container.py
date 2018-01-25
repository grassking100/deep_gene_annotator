"""This submodule provide abstract class about model container"""
from abc import ABCMeta, abstractmethod
class SizeIncosistentException(Exception):
    """Error about size incosistent exception"""
    pass
class Container(metaclass=ABCMeta):
    """The class hold model and handle with data input, model pedict and model result"""
    def __init__(self):
        self._model = None
        self._data = {}
        self._result = {}
    @property
    def result(self):
        """Get result"""
        if self._result is  None:
            raise Exception("There is not result yet")
        return self._result
    @result.setter
    def result(self, result):
        """Set result"""
        self._result = result
    @abstractmethod
    def _validate_required(self):
        """Abstract method for validate required situation is met"""
        pass
    def clean_result(self):
        """clean result"""
        self.result = {}
    def load_data(self, data, kind):
        """Check for its validity and load data to container"""
        self._validate_key(kind)
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
    def _validate_key(self, key):
        """Validate the key """
        pass
    def _valid_data_shape(self, input_sequencse, output_sequences):
        """Validate if length of input and output sequence is same"""
        if input_sequencse.shape[1] != output_sequences.shape[1]:
            raise SizeIncosistentException("Input data must have same size of "+
                                           "output data,"+
                                           "size of input data is "+str(input_sequencse.shape[1])+
                                           ",and size of output data is "+
                                           str(output_sequences.shape[1]))
