"""A submodule about metric"""
from ..genome_handler.utils import process_tensor
from ..utils.exception import LengthNotEqualException
from abc import ABCMeta,abstractmethod
import tensorflow as tf
import keras.backend as K

class Metric(metaclass=ABCMeta):
    def __init__(self,name='Metric'):
        self._name = name
        self._default_method = None
    @property
    def name(self):
        return self._name
    @property
    def __name__(self):
        return self._name
    @property
    def default_method(self):
        return self._default_method
    @default_method.setter
    def default_method(self, value):
        self._default_method = value
        
class BatchCounter(Metric):
    def __init__(self,name='BatchCounter'):
        super().__init__(name=name)
        self._constant = 1
        self._default_method = None
    def set_constant(self, value):
        self._constant = value
    def get_constant(self):
        return K.variable(value=self._constant)

class SeqAnnMetric(Metric):
    def __init__(self, name='SeqAnnMetric', values_to_ignore=None):
        super().__init__(name=name)
        self._values_to_ignore = values_to_ignore
        self._pred = None
        self._true = None
    def _get_preprocessed(self, true, pred):
        clean_true, clean_pred = process_tensor(true, pred,values_to_ignore=self._values_to_ignore)  
        return (clean_true,clean_pred)
    def set_data(self, true, pred):
        true, pred = self._get_preprocessed(true, pred)
        self._true = true
        self._pred = pred

class SpecificTypeMetric(SeqAnnMetric):
    def __init__(self,target_index,name='SpecificTypeMetric', values_to_ignore=None):
        super().__init__(name=name, values_to_ignore=values_to_ignore)
        self._target_index = target_index
    def get_true_positive(self):
        return tf.cast(tf.reduce_sum(tf.multiply(self._true, self._pred)), tf.int64)
    def get_true_negative(self):
        return tf.cast(tf.reduce_sum(tf.multiply(1-self._true, 1-self._pred)), tf.int64)
    def get_false_negative(self):
        return tf.cast(tf.reduce_sum(tf.multiply(self._true, 1-self._pred)), tf.int64)
    def get_false_positive(self):
        return tf.cast(tf.reduce_sum(tf.multiply(1-self._true, self._pred)), tf.int64)
    def _get_negative(self, data):
        return tf.count_nonzero(1-data)
    def _get_positive(self, data):
        return tf.count_nonzero(data)
    def get_real_length(self):
        return tf.size(self._true)
    def get_pred_length(self):
        return tf.size(self._pred)
    def get_real(self):
        return self._true
    def get_pred(self):
        return self._pred
    def _get_binary(self, data):
        return tf.cast(tf.equal(tf.argmax(data,1),self._target_index),tf.int64)
    def _get_preprocessed(self,true,pred):
        clean_true, clean_pred = super()._get_preprocessed(true,pred)
        binary_true = self._get_binary(clean_true)
        binary_pred = self._get_binary(clean_pred)
        return (binary_true,binary_pred)

class Accuracy(SeqAnnMetric):
    def __init__(self, name=None, values_to_ignore=None,type_="categorical_accuracy"):
        super().__init__(name=name or type_,values_to_ignore=values_to_ignore)
        try:
            exec('from keras.metrics import {type_}'.format(type_=type_))
            exec('self._acc_function={type_}'.format(type_=type_))
        except ImportError as e:
            raise Exception(accuracy_type + ' cannot be found in keras.metrics')
    def get_result(self):
        accuracy = tf.reduce_mean(self._acc_function(self._true, self._pred))
        return accuracy
    def __call__(self, y_true, y_pred):
        self.set_data(y_true, y_pred)
        return self.get_result()