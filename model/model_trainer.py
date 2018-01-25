"""This submodule provides trainer to train model"""
from keras.callbacks import TensorBoard
from . import numpy
from . import Container
class ModelTrainer(Container):
    """a trainer which will train and evaluate the model"""
    def __init__(self):
        super().__init__()
        self.__valid_key = ['train_x','train_y','validation_x','validation_y']
    def _validate_key(self, key):
        """Validate the key"""
        if key not in self.__valid_key:
            raise Exception("Key:"+key+" is not a valid key")
    def _validate_required(self):
        """Validate required data"""
        attrs = ['_model','_data']
        for attr in attrs:
            if getattr(self,attr) is None:
                raise Exception("ModelTrainer needs "+attr+" to complete the quest")
        for key in self.__valid_key:
            if key not in self._data.keys():
                raise Exception("ModelTrainer needs data about "+key+" to complete the quest")
        self._valid_data_shape(self._data['train_x'],self._data['train_y'])
        self._valid_data_shape(self._data['validation_x'],self._data['validation_y'])
    def train(self, epoches, batch_size, shuffle, verbose, log_file):
        """Train model"""
        self._validate_required()
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
