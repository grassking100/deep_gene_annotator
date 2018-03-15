"""This submodule help user to create metric function,or customize objects or cnn setting"""
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
import tensorflow as tf
from . import process_tensor
from . import Builder
from . import rename

class CategoricalCrossentropyFactory:
    """This class create and return categorical cross entropy function"""
    def create(self, weights=None, values_to_ignore=None, name="loss"):
        """return cross entropy function"""
        @rename(name)
        def crossentropy(y_true, y_pred):
            """calculate static categorical cross entropy between y_true and y_pred"""
            if values_to_ignore is not None:
                (y_true, y_pred) = process_tensor(y_true, y_pred,
                                                  values_to_ignore)
            if weights is not None:
                y_true = tf.multiply(y_true, weights)
            loss = tf.reduce_mean(categorical_crossentropy(y_true, y_pred))
            return loss
        return crossentropy

class CategoricalAccuracyFactory:
    """This class create and return categorical accuracy function"""
    def create(self, values_to_ignore=None, name="accuracy"):
        """return accuracy function"""
        @rename(name)
        def advanced_categorical_accuracy(y_true, y_pred):
            """calculate categorical accuracy"""
            (y_true, y_pred) = process_tensor(y_true, y_pred,values_to_ignore)
            accuracy = tf.reduce_mean(categorical_accuracy(y_true, y_pred))
            return accuracy
        return advanced_categorical_accuracy
class CnnSettingBuilder(Builder):
    """A class which generate setting about multiple convolution layer"""
    def __init__(self):
        super().__init__()
        self._layers_settings = []
    def clean_layers(self):
        """reset setting"""
        self._layers_settings = []
        return self
    def add_layer(self, filter_num, filter_size):
        """Add a convolution layer setting"""
        setting = {'filter_num':filter_num, 'filter_size':filter_size}
        self._layers_settings.append(setting)
        return self
    def _validate(self):
        pass
    def build(self):
        """Get settings"""
        self._validate()
        return self._layers_settings
