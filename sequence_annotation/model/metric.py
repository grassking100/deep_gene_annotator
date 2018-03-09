from . import remove_terminal
from abc import ABCMeta
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
import tensorflow as tf
class Metric(metaclass=ABCMeta):
    def __init__(self,name):
        self._name=name
    @property
    def name(self):
        return self._name
    @property
    def __name__(self):
        return self._name
    def __call__(self, y_true, y_pred):
        """Update the state at the completion of each batch.
        # Arguments
            y_true: the batch_wise labels
            y_pred: the batch_wise predictions
        # Returns
            The calculated data
        """
        pass
class CategoricalMetric(Metric):
    def __init__(self, name, terminal_signal=None):
        super().__init__(name)
        self._terminal_signal = terminal_signal
        self._pred = None
        self._true = None
    def _get_preprocessed(self,true,pred):
        clean_true, clean_pred = remove_terminal(true, pred,self._terminal_signal)  
        return (clean_true,clean_pred)
    def get_true_positive(self):
        return tf.cast(tf.reduce_sum(tf.multiply(self._true, self._pred)), tf.int64)
    def get_true_negative(self):
        return tf.cast(tf.reduce_sum(tf.multiply(1-self._true, 1-self._pred)), tf.int64)
    def get_false_negative(self):
        return tf.cast(tf.reduce_sum(tf.multiply(self._true, 1-self._pred)), tf.int64)
    def get_false_positive(self):
        return tf.cast(tf.reduce_sum(tf.multiply(1-self._true, self._pred)), tf.int64)
    def get_negative(self,data):
        return tf.count_nonzero(1-data)
    def get_positive(self,data):
        return tf.count_nonzero(data)
    def set_data(self,true,pred):
        true, pred = self._get_preprocessed(true, pred)
        self._true=true
        self._pred=pred
class SpecificTypeMetric(CategoricalMetric):
    def __init__(self, name, target_index,terminal_signal=None):
        super().__init__(name,terminal_signal)
        self._target_index = target_index
    def get_numeric(self, data):
        return tf.cast(tf.equal(tf.argmax(data, 0),self._target_index),tf.int64)
    def _get_preprocessed(self,true,pred):
        clean_true, clean_pred = super()._get_preprocessed(true,pred)
        numeric_true = self.get_numeric(clean_true)
        numeric_pred = self.get_numeric(clean_pred)
        return (numeric_true,numeric_pred)
class StaticCrossEntropy(CategoricalMetric):
    def __init__(self, name, class_number,
                 weights,terminal_signal=None):
        super().__init__(name,class_number,terminal_signal)
        self._weights = weights
    def get_crossentropy(self):
        if self._weights is not None:
            true = tf.multiply(self._true, self._weights)
        loss = tf.reduce_mean(categorical_crossentropy(true, self._pred))
        return loss
    def __call__(self, y_true, y_pred):
        self.set_data(y_true, y_pred)
        return self.get_crossentropy()
class CategoricalAccuracy(CategoricalMetric):
    def get_accuracy(self):
        accuracy = tf.reduce_mean(categorical_accuracy(self._true, self._pred))
        return accuracy
    def __call__(self, y_true, y_pred):
        self.set_data(y_true, y_pred)
        return self.get_accuracy()
class CategoricalPrecision(SpecificTypeMetric):
    def get_precision(self):
        true_positive = self.get_true_positive()
        false_positive = self.get_false_positive()
        return  true_positive/(true_positive+false_positive)
    def __call__(self, y_true, y_pred):
        self.set_data(y_true, y_pred)
        return  self.get_precision()
class CategoricalRecall(SpecificTypeMetric):
    def get_recall(self):
        true_positive = self.get_true_positive()
        false_negative = self.get_false_negative()
        return  true_positive/(true_positive+false_negative)
    def __call__(self, y_true, y_pred):    
        self.set_data(y_true, y_pred)
        return self.get_recall()
class CategoricalMetricFactory(metaclass=ABCMeta):
    def create(self, type_, name, weights=None, terminal_signal=None, target_index=None):
        if type_=="accuracy":
            metric = CategoricalAccuracy(name,terminal_signal)
        elif type_=="static_crossentropy":
            metric = StaticCrossEntropy(name,weights,terminal_signal)
        elif type_=="specific_type":
            metric = SpecificTypeMetric(name, target_index,terminal_signal=None)
        elif type_=="precision":
            metric = CategoricalPrecision(name,target_index,terminal_signal)
        elif type_=="recall":
            metric = CategoricalRecall(name,target_index,terminal_signal)
        else:
            raise Exception(type_+" is not correct metric type")
        return metric