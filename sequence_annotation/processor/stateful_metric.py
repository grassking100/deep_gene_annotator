"""A submodule about metric layer"""
from keras.engine.topology import Layer
from abc import ABCMeta
from keras import backend as K
import tensorflow as tf

class StatefulMetric(Layer,metaclass=ABCMeta):
    """A layer can be called to calculate stateful metric"""
    def __init__(self, metric, name=None):
        super().__init__()
        self._metric = metric
        self._data = K.variable(value=0, dtype='float32')
        self.name = name or metric.name
        self.stateful = True
    def reset_states(self):
        """
            Reset the state at the beginning of 
            training and evaluation for each epoch.
        """
        K.set_value(self._data, 0)
    def _calculate(self, y_true, y_pred):
        """Calculate and return result of metric"""
        return self._metric(y_true, y_pred)
    def __call__(self, y_true, y_pred):
        """Make layer callable"""
        result = tf.cast(self._calculate(y_true,y_pred),dtype='float32')
        self.add_update(K.update_add(self._data, result),inputs=[y_true, y_pred])
        return self._data
