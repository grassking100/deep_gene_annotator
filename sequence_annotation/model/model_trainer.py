"""This submodule provides trainer to train model"""
from keras.callbacks import TensorBoard
import numpy
from . import ModelWorker
from . import DataValidator, DictValidator, AttrValidator
class ModelTrainer(ModelWorker):
    """a trainer which will train and evaluate the model"""
    def __init__(self):
        super().__init__()
        self._valid_data_keys = ['train_x','train_y','validation_x','validation_y']
    def _validate(self):
        """Validate required data"""
        attr_validator = AttrValidator(self)
        attr_validator.is_protected_validated = True
        attr_validator.validate()
        dict_validator = DictValidator(self._data)
        dict_validator.key_of_validated_value = self._valid_data_keys
        dict_validator.key_must_included = self._valid_data_keys
        dict_validator.invalid_values = [None]
        dict_validator.validate()
        data_validator = DataValidator()
        data_validator.same_shape(self._data['train_x'],self._data['train_y'])
        data_validator.same_shape(self._data['validation_x'],self._data['validation_y'])
    def train(self, epoches, batch_size, shuffle, verbose, log_file):
        """Train model"""
        self._validate()
        tb_call_back = TensorBoard(log_dir='./'+log_file, histogram_freq=1,
                                   write_graph=True, write_grads=True,
                                   write_images=True)
        tb_call_back.set_model(self.model)
        #training and evaluating the model
        history = self.model.fit(numpy.array(self._data['train_x']),
                                 numpy.array(self._data['train_y']),
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 epochs=epoches,
                                 verbose=verbose,
                                 validation_data=(numpy.array(self._data['validation_x']),
                                                  numpy.array(self._data['validation_y'])),
                                 callbacks=[tb_call_back])
        #add record to histories
        for key, value in history.history.items():
            if key in self.result.keys():
                self.result[key] += value
            else:
                self.result[key] = value
        return self
