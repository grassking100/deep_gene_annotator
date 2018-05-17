"""This submodule help user to create metric function,or customize objects or cnn setting"""
import warnings
import tensorflow as tf
from keras.metrics import categorical_accuracy
from . import SeqAnnDataHandler
from . import Builder
from . import rename

class LossFactory():
    """This class create and return loss function"""
    def _reversed_count_weight(self,seq_tensor):
        dim = tf.cast(tf.shape(seq_tensor)[2], tf.float32)
        class_count = tf.cast(tf.reduce_sum(seq_tensor,[0,1]), tf.float32)
        reversed_count = tf.divide(1,(class_count+1))
        reversed_count_sum = tf.reduce_sum(reversed_count)
        weight = tf.divide(tf.multiply(1.0,reversed_count),reversed_count_sum)
        return tf.multiply(weight,dim)
    def create(self, weight=None, values_to_ignore=None, name="loss",
               loss_type="categorical_crossentropy",dynamic_weight_method=None):
        """return loss function"""
        @rename(name)
        def crossentropy(y_true, y_pred):
            """calculate loss between y_true and y_pred"""
            if values_to_ignore is not None:
                (y_true, y_pred) = SeqAnnDataHandler.process_tensor(y_true, y_pred,
                                                                    values_to_ignore=values_to_ignore)
            if weight or dynamic_weight_method is not None:
                if loss_type in ['categorical_crossentropy','binary_crossentropy']:
                    if dynamic_weight_method is not None:
                        warnings.warn("Weight will be recalucalated by dynamic weight")
                        if dynamic_weight_method=="reversed_count_weight":
                            weight_ = self._reversed_count_weight(y_true)
                        else:
                            raise Exception(dynamic_weight_method+"is not be implemented yet")
                    else:
                        weight_ = list(weight)
                    y_true = tf.multiply(y_true, weight_)
                else:
                    raise Exception("Weight loss is not support for "+loss_type+" yet.")
            try:
                exec('from keras.losses import {loss_type}'.format(loss_type=loss_type))
                exec('self._loss_function={loss_type}'.format(loss_type=loss_type))
                loss = tf.reduce_mean(self._loss_function(y_true, y_pred))
                return loss
            except ImportError as e:
                raise Exception(loss_type+' cannot be found in keras.losses')
        return crossentropy

class CategoricalAccuracyFactory:
    """This class create and return categorical accuracy function"""
    def create(self, values_to_ignore=None, name="accuracy"):
        """return accuracy function"""
        @rename(name)
        def advanced_categorical_accuracy(y_true, y_pred):
            """calculate categorical accuracy"""
            (y_true, y_pred) = SeqAnnDataHandler.process_tensor(y_true, y_pred,
                                                                values_to_ignore=values_to_ignore)
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
