"""A submodule about metric"""
from abc import ABCMeta,abstractmethod
import tensorflow as tf
import keras.backend as K

class Metric(metaclass=ABCMeta):
    def __init__(self,name,values_to_ignore=None):
        self.name = name
        self._values_to_ignore = values_to_ignore
        self._dtype = None
    @property
    def dtype(self):
        return self._dtype
    @abstractmethod    
    def __call__(self, y_true, y_pred):
        pass

class BatchCount(Metric):
    def __init__(self,name='batch_count',values_to_ignore=None):
        super().__init__(name,values_to_ignore)
        self._dtype = 'int32'

    def __call__(self, true, pred):
        return K.variable(value=1)

class SampleCount(Metric):
    def __init__(self, name='sample_count',values_to_ignore=None):
        super().__init__(name,values_to_ignore=values_to_ignore)
        self._dtype = 'int32'

    def __call__(self, true, pred):
        return  tf.shape(true)[0]
    
class SeqAnnMetric(Metric):
    def __init__(self, name,values_to_ignore):
        super().__init__(name,values_to_ignore)
        self._pred = None
        self._true = None
        self._mask = None

    def _set_data(self, true, pred):
        self._true = true
        self._pred = pred

    def _set_mask(self, true):
        self._mask = K.any(K.not_equal(true, self._values_to_ignore), axis=-1)

class BinaryMetric(SeqAnnMetric):
    def __init__(self,target_index,name, values_to_ignore=None):
        super().__init__(name, values_to_ignore)
        self._target_index = target_index
        self._dtype = 'int32'
        
    def _get_binary(self, data):
        return tf.cast(tf.equal(tf.argmax(data,2),self._target_index),self._dtype)

    def _set_data(self, true, pred):
        binary_true = self._get_binary(true)
        binary_pred = self._get_binary(pred)
        super()._set_data(binary_true,binary_pred)

    def __call__(self, y_true, y_pred):
        if self._values_to_ignore is not None:
            self._set_mask(y_true)
        self._set_data(y_true, y_pred)
        result = self._get_result()
        if self._mask is not None:
            result *= tf.cast(self._mask,self._dtype)
        result = K.sum((result))
        return result

class TruePositive(BinaryMetric):
    def __init__(self, target_index, name='TP',values_to_ignore=None):
        super().__init__(target_index, name=name,values_to_ignore=values_to_ignore)

    def _get_result(self):
        result = tf.multiply(self._true, self._pred)
        return result

class TrueNegative(BinaryMetric):
    def __init__(self, target_index, name='TN',values_to_ignore=None):
        super().__init__(target_index, name=name,values_to_ignore=values_to_ignore)

    def _get_result(self):
        return tf.multiply(1-self._true, 1-self._pred)

class FalsePositive(BinaryMetric):
    def __init__(self, target_index, name='FP',values_to_ignore=None):
        super().__init__(target_index, name=name,values_to_ignore=values_to_ignore)

    def _get_result(self):
        return tf.multiply(1-self._true, self._pred)
    
class FalseNegative(BinaryMetric):
    def __init__(self, target_index, name='FN',values_to_ignore=None):
        super().__init__(target_index, name=name,values_to_ignore=values_to_ignore)

    def _get_result(self):
        return tf.multiply(self._true, 1-self._pred)

class Accuracy(SeqAnnMetric):
    def __init__(self, name=None, values_to_ignore=None, type_="categorical_accuracy"):
        super().__init__(name or type_,values_to_ignore)
        self._dtype = 'float32'
        try:
            exec('from keras.metrics import {type_}'.format(type_=type_))
            exec('self._acc_function={type_}'.format(type_=type_))
        except ImportError:
            raise Exception(accuracy_type + ' cannot be found in keras.metrics')

    def _get_result(self):
        accuracy = self._acc_function(self._true, self._pred)
        return accuracy

    def __call__(self, y_true, y_pred):
        if self._values_to_ignore is not None:
            self._set_mask(y_true)
        self._set_data(y_true, y_pred)
        result = self._get_result()
        if self._mask is not None:
            result *= tf.cast(self._mask,self._dtype)
            result = result / tf.cast(K.sum(self._mask),'int32')
        else:
            result = K.mean(result)
        return result    


"""Alias"""
TP = TruePositive
TN = TrueNegative
FP = FalsePositive
FN = FalseNegative