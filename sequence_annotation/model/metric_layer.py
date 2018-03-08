from keras.engine.topology import Layer
from abc import abstractmethod,ABCMeta
from . import removed_terminal_tensors
from keras import backend as K
class MetricLayer(Layer,metaclass=ABCMeta):
    def __init__(self, classified_type, specific_type_metric):
        super().__init__()
        self._classified_type = classified_type
        self._metric = specific_type_metric
        self._data = K.variable(value=0, dtype='int64')
        self.name = specific_type_metric.name
        print("Is layer:"+str(isinstance (self,Layer)))
    @property
    def classified_type(self):
        return self._classified_type
    def reset_states(self):
        """Reset the state at the beginning of training and evaluation for each
         epoch.
        """
        print("Reset layer:"+self.name)
        K.set_value(self._data, 0)
        print("Reset layer:"+self._data)
class TruePositive(MetricLayer):
    def __call__(self, y_true, y_pred):
        self._metric.set_data(y_true, y_pred)
        true_positive = self._metric.get_true_positive()
        updated = self._data + true_positive
        self.add_update(K.update_add(self._data, true_positive),inputs=[y_true, y_pred])
        return updated
class TrueNegative(MetricLayer):
    def __call__(self, y_true, y_pred):
        self._metric.set_data(y_true, y_pred)
        true_negative = self._metric.get_true_negative()
        updated = self._data + true_negative
        self.add_update(K.update_add(self._data, true_negative),inputs=[y_true, y_pred])
        return updated
class FalseNegative(MetricLayer):
    def __call__(self, y_true, y_pred):
        self._metric.set_data(y_true, y_pred)
        false_negative = self._metric.get_false_negative()
        updated = self._data + false_negative
        self.add_update(K.update_add(self._data, false_negative),inputs=[y_true, y_pred])
        return updated
class FalsePositive(MetricLayer):
    def __call__(self, y_true, y_pred):
        self._metric.set_data(y_true, y_pred)
        false_positive = self._metric.get_false_positive()
        updated = self._data + false_positive
        self.add_update(K.update_add(self._data, false_positive),inputs=[y_true, y_pred])
        return updated