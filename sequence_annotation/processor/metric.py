"""A submodule about metric"""
from ..utils.utils import process_tensor
from ..utils.exception import LengthNotEqualException
from abc import ABCMeta,abstractmethod
import tensorflow as tf
import keras.backend as K

class Metric(metaclass=ABCMeta):
    def __init__(self,name,values_to_ignore):
        self.name = name
        self._values_to_ignore = values_to_ignore
    @abstractmethod    
    def __call__(self, y_true, y_pred):
        pass
class BatchCount(Metric):
    def __init__(self,name='batch_count',values_to_ignore=None):
        super().__init__(name,values_to_ignore)
        self._constant = 1
    def __call__(self, true, pred):
        return K.variable(value=1)

class SeqAnnMetric(Metric):
    def __init__(self, name,values_to_ignore):
        super().__init__(name,values_to_ignore)
        self._pred = None
        self._true = None
    def _set_data(self, true, pred):
        self._true = true
        self._pred = pred
    @abstractmethod
    def _get_result(self):
        pass
    def __call__(self, y_true, y_pred):
        self._set_data(y_true, y_pred)
        return self._get_result()

class BinaryMetric(SeqAnnMetric):
    def __init__(self,target_index,name, values_to_ignore=None):
        super().__init__(name, values_to_ignore)
        self._target_index = target_index
    def _get_binary(self, data):
        return tf.cast(tf.equal(tf.argmax(data,1),self._target_index),tf.int64)
    def _set_data(self, true, pred):
        binary_true = self._get_binary(true)
        binary_pred = self._get_binary(pred)
        super()._set_data(binary_true,binary_pred)

class SampleCount(SeqAnnMetric):
    def __init__(self, name='count',values_to_ignore=None):
        super().__init__(name,values_to_ignore)
    def _get_result(self):
        return tf.shape(self._true)[0]

class TruePositive(BinaryMetric):
    def __init__(self, target_index, name='TP'):
        super().__init__(target_index, name=name)
    def _get_result(self):
        return tf.reduce_sum(tf.multiply(self._true, self._pred))

class TrueNegative(BinaryMetric):
    def __init__(self, target_index, name='TN'):
        super().__init__(target_index, name=name)
    def _get_result(self):
        return tf.reduce_sum(tf.multiply(1-self._true, 1-self._pred))

class FalsePositive(BinaryMetric):
    def __init__(self, target_index, name='FP'):
        super().__init__(target_index, name=name)
    def _get_result(self):
        return tf.reduce_sum(tf.multiply(1-self._true, self._pred))
    
class FalseNegative(BinaryMetric):
    def __init__(self, target_index, name='FN'):
        super().__init__(target_index, name=name)
    def _get_result(self):
        return tf.reduce_sum(tf.multiply(self._true, 1-self._pred))

class Accuracy(SeqAnnMetric):
    def __init__(self, name=None, values_to_ignore=None, type_="categorical_accuracy"):
        super().__init__(name or type_,values_to_ignore)
        try:
            exec('from keras.metrics import {type_}'.format(type_=type_))
            exec('self._acc_function={type_}'.format(type_=type_))
        except ImportError as e:
            raise Exception(accuracy_type + ' cannot be found in keras.metrics')
    def _get_result(self):
        accuracy = tf.reduce_mean(self._acc_function(self._true, self._pred))
        return accuracy
    def __call__(self, y_true, y_pred):
        self.set_data(y_true, y_pred)
        return self._get_result()

"""Alias"""
TP = TruePositive
TN = TrueNegative
FP = FalsePositive
FN = FalseNegative