import numpy as np
from abc import ABCMeta, abstractmethod
from ..utils.metric import MetricCalculator, get_categorical_metric, get_confusion_matrix

class ICallback(metaclass=ABCMeta):
    @abstractmethod
    def get_config(self):
        pass

    def on_work_begin(self, worker):
        pass

    def on_work_end(self):
        pass

    def on_epoch_begin(self, counter):
        pass

    def on_epoch_end(self, metric):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self, seq_data, masks, predicts, metric, outputs):
        pass

    def get_data(self):
        return {}
    
    
def get_prefix(prefix=None):
    if prefix is not None and len(prefix) > 0:
        prefix += "_"
    else:
        prefix = ""
    return prefix


class Callback(ICallback):
    def get_config(self, **kwargs):
        config = {}
        return config


class Callbacks(ICallback):
    def __init__(self, callbacks=None):
        self._callbacks = []
        if callbacks is not None:
            self.add(callbacks)

    def get_config(self, **kwargs):
        config = {}
        for callback in self._callbacks:
            name = callback.__class__.__name__
            config[name] = callback.get_config()
        return config

    def clean(self):
        self._callbacks = []
        return self

    @property
    def callbacks(self):
        return self._callbacks

    def add(self, callbacks):
        list_ = []
        if isinstance(callbacks, Callbacks):
            for callback in callbacks.callbacks:
                if isinstance(callback, Callbacks):
                    list_ += callback.callbacks
                else:
                    list_ += [callback]
        elif isinstance(callbacks, list):
            list_ += callbacks
        else:
            list_ += [callbacks]
        self._callbacks += list_
        return self

    def on_work_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_work_begin(**kwargs)

    def on_work_end(self):
        for callback in self.callbacks:
            callback.on_work_end()

    def on_epoch_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(**kwargs)

    def on_batch_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(**kwargs)

    def on_batch_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(**kwargs)

    def get_data(self):
        record = super().get_data()
        for callback in self.callbacks:
            #if hasattr(callback, 'get_data'):
            for type_, value in callback.get_data().items():
                record[type_] = value
        return record

class DataCallback(Callback):
    def __init__(self, prefix=None):
        self._prefix = get_prefix(prefix)
        self._data = None

    def _reset(self):
        pass

    def on_work_begin(self, **kwargs):
        self._reset()
        
    def get_config(self):
        config = super().get_config()
        config['prefix'] = self._prefix
        return config


class MeanRecorder(DataCallback):
    def __init__(self, prefix=None):
        super().__init__(prefix)
        self._batch_count = None
        self.round_value = 5

    def on_epoch_begin(self, **kwargs):
        self._reset()

    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['round_value'] = self.round_value
        return config

    def _reset(self):
        self._data = {}
        self._batch_count = 0

    def on_batch_end(self, metric, **kwargs):
        if self._batch_count == 0:
            for key in metric.keys():
                self._data[key] = 0
        for key, value in metric.items():
            self._data[key] += value
        self._batch_count += 1

    def get_data(self):
        data = super().get_data()
        for key, value in self._data.items():
            value = round(value / self._batch_count, self.round_value)
            data[self._prefix + key] = value
        data[self._prefix + 'batch_count'] = self._batch_count
        return data


class DataHolder(DataCallback):
    def __init__(self, prefix=None):
        super().__init__(prefix)
        self.round_value = 5

    def on_epoch_begin(self, **kwargs):
        self._reset()

    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['round_value'] = self.round_value
        return config

    def _reset(self):
        self._data = {}

    def on_batch_end(self, metric, **kwargs):
        self._data.update(metric)

    def get_data(self):
        data = super().get_data()
        for key, value in self._data.items():
            value = round(value, self.round_value)
            data[self._prefix + key] = value
        return data


class CategoricalMetric(DataCallback):
    def __init__(self, label_num=None, label_names=None, prefix=None,round_value=None):
        super().__init__(prefix)
        self._label_num = label_num or 3
        self._label_names = label_names
        self._metric_calculator = MetricCalculator(self._label_num,prefix=self._prefix,
                                                  label_names=self._label_names)
        self._result = {}
        self._round_value = round_value or 5

    def on_epoch_begin(self, **kwargs):
        self._reset()

    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['label_num'] = self._label_num
        config['label_names'] = self._label_names
        config['round_value'] = self._round_value
        return config
    
    def _reset(self):
        self._data = {}
        for type_ in ['TP', 'FP', 'FN']:
            self._data[type_] = [0] * self._label_num
    
    def on_batch_end(self, seq_data, masks, predicts, **kwargs):
        predicted_anns = predicts.get('annotation').cpu().numpy()
        answers = seq_data.get('answer').cpu().numpy()
        masks = masks.cpu().numpy()
        # N,C,L
        data = get_categorical_metric(predicted_anns,answers,masks)
        for type_ in ['TP', 'FP', 'FN']:
            for index in range(self._label_num):
                self._data[type_][index] += data[type_][index]
        self._result = self._metric_calculator(self._data)

    def get_data(self):
        return self._result

class ConfusionMatrix(DataCallback):
    def __init__(self, label_num=None, prefix=None):
        super().__init__(prefix)
        self._label_num = label_num or 3
        self._result = {}

    def on_epoch_begin(self, **kwargs):
        self._reset()

    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['label_num'] = self._label_num
        return config

    def on_batch_end(self, seq_data, masks, predicts, **kwargs):
        predicted_anns = predicts.get('annotation').cpu().numpy()
        answers = seq_data.get('answer').cpu().numpy()
        # N,C,L
        data = get_confusion_matrix(predicted_anns,answers,masks.cpu().numpy())
        self._data += np.array(data)
        self._result = {self._prefix + 'confusion_matrix': self._data.tolist()}

    def get_data(self):
        return self._result

    def _reset(self):
        self._data = np.array([[0] * self._label_num] * self._label_num)
