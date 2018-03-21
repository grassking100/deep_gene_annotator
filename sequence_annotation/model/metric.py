"""A submodule about metric"""
from . import SeqAnnDataHandler
from . import LengthNotEqualException
from abc import ABCMeta
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
import tensorflow as tf
import keras.backend as K
class Metric(metaclass=ABCMeta):
    def __init__(self,name):
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
class DependentMetric(Metric):
    def __init__(self,name):
        super().__init__(name=name)
        self._constant = 100
        self._default_method = None
    def set_constant(self, value):
        self._constant = value
    def get_constant(self):
        return K.variable(value=self._constant)
class CategoricalMetric(Metric):
    def __init__(self, name, values_to_ignore=None):
        super().__init__(name=name)
        self._values_to_ignore = values_to_ignore
        self._pred = None
        self._true = None
    def _get_preprocessed(self, true, pred):
        clean_true, clean_pred = SeqAnnDataHandler.process_tensor(true, pred,values_to_ignore=self._values_to_ignore)  
        return (clean_true,clean_pred)
    def _validate_input(self, true, pred):
        true_shape = K.int_shape(true)[0:2]
        pred_shape = K.int_shape(pred)[0:2]
        if true_shape != pred_shape:
            raise LengthNotEqualException(true_shape, pred_shape)
    def set_data(self, true, pred):
        self._validate_input(true, pred)
        true, pred = self._get_preprocessed(true, pred)
        self._true = true
        self._pred = pred
    """def get_config(self):
        config = super().get_config()
        config['values_to_ignore'] = self._values_to_ignore
        return config"""
class SpecificTypeMetric(CategoricalMetric):
    def __init__(self, name, target_index, values_to_ignore=None):
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
class StaticCrossentropy(CategoricalMetric):
    def __init__(self, name,weights,values_to_ignore=None):
        super().__init__(name=name,values_to_ignore=values_to_ignore)
        self._weights = weights
    def get_result(self):
        if self._weights is not None:
            true = tf.multiply(self._true, self._weights)
        else:
            true = self._true
        loss = tf.reduce_mean(categorical_crossentropy(true, self._pred))
        return loss
    def __call__(self, y_true, y_pred):
        self.set_data(y_true, y_pred)
        return self.get_result()
class CategoricalAccuracy(CategoricalMetric):
    def get_result(self):
        accuracy = tf.reduce_mean(categorical_accuracy(self._true, self._pred))
        return accuracy
    def __call__(self, y_true, y_pred):
        self.set_data(y_true, y_pred)
        return self.get_result()
class MetricFactory(metaclass=ABCMeta):
    """A factory creates metric"""
    def create(self, type_, name, weights=None, values_to_ignore=None, target_index=None):
        if type_ == "accuracy":
            metric = CategoricalAccuracy(name=name,values_to_ignore=values_to_ignore)
        elif type_ == "static_crossentropy":
            metric = StaticCrossentropy(name=name, weights=weights,
                                        values_to_ignore=values_to_ignore)
        elif type_ == "specific_type":
            metric = SpecificTypeMetric(name=name, target_index=target_index,
                                        values_to_ignore=values_to_ignore)
        elif type_ == "dependent":
            metric = DependentMetric(name=name)
        else:
            raise Exception(type_ + " is not correct metric type")
        return metric