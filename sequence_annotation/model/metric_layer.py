"""A submodule about metric layer"""
from keras.engine.topology import Layer
from abc import ABCMeta
from keras import backend as K
import tensorflow as tf
class MetricLayer(Layer,metaclass=ABCMeta):
    """A layer can be called to calculate metric"""
    def __init__(self, metric, metric_method, name=None, class_type=None):
        super().__init__()
        self._class_type = class_type
        self._metric = metric
        self._data = K.variable(value=0, dtype='int64')
        self.name = name or metric.name+"_layer"
        self.stateful = True
        if hasattr(self._metric, metric_method.__name__):
            self._metric.default_method = metric_method
        else:
            mess = "Method,{method}, doesn't exist in metric,{metric}."
            mess = mess.format(method=metric_method,metric=self._metric)
            raise Exception(mess)
    @property
    def class_type(self):
        return self._class_type
    def reset_states(self):
        """
            Reset the state at the beginning of 
            training and evaluation for each epoch.
        """
        #print("Reset layer:"+self.name)
        K.set_value(self._data, 0)
    def _calculate(self, y_true, y_pred):
        """Calculate and return result of metric"""
        if hasattr(self._metric,"set_data"):
            self._metric.set_data(y_true, y_pred)
        if self._metric.default_method is None:
            mess = "Default method of {metric} is not assigned"
            raise Exception(mess.format(metric=self._metric))
        else:
            return self._metric.default_method()
    def __call__(self, y_true, y_pred):
        """Make layer callable"""
        result = tf.cast(self._calculate(y_true, y_pred), tf.int64)
        self.add_update(K.update_add(self._data, result),inputs=[y_true, y_pred])
        return self._data
class MetricLayerFactory(metaclass=ABCMeta):
    """A Factory creates metric layer"""
    def _get_method(self,metric,method_name):
        """Return metric method if exist"""
        if hasattr(metric,method_name):
            return getattr(metric,method_name)
        else:
            mess = "Method,{method}, doesn't exist in metric,{metric}."
            mess = mess.format(method=method_name,metric=metric)
            raise Exception(mess)
    def create(self, method_type, metric, class_type = None):
        """Create metric layer"""
        if method_type=="TP":
            metric_name ='get_true_positive'
        elif method_type=="TN":
            metric_name = 'get_true_negative'
        elif method_type=="FP":
            metric_name = 'get_false_positive'
        elif method_type=="FN":
            metric_name = 'get_false_negative'
        elif method_type=="RL":
            metric_name = 'get_real_length'
        elif method_type=="PL":
            metric_name = 'get_pred_length'
        elif method_type=="Constant":
            metric_name = 'get_constant'
        else:
            raise Exception(method_type+" is not correct method")
        metric_method = self._get_method(metric,metric_name)
        metric = MetricLayer(metric=metric,
                             metric_method=metric_method,
                             class_type=class_type)
        return metric