"""This submodule provides trainer to train model"""
import tensorflow as tf
import keras.backend as K
import json
import numpy as np
from . import Worker
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

class TestWorker(Worker):
    def __init__(self, path_root, batch_size, model,data,
                 previous_result=None,
                 use_generator=False):
        super().__init__(path_root)
        self._model = model
        self._data = data
        self._result = {}
        self._batch_size = batch_size
        self._use_generator = use_generator
    def before_work(self):
        pass
    def after_work(self):
        with open(self._path_root + '/result.json', 'w') as outfile:  
            json.dump(self._result, outfile)
    def work(self):
        train_x = self._data['training']['inputs']
        train_y = self._data['training']['answers']
        history = []
        verbose =int(self.is_verbose_visible)
        if self._batch_size==1:
            for x,y in zip(train_x,train_y):
                history.append(self._model.evaluate(np.array([x]),np.array([y]),
                                                    batch_size=1,
                                                    verbose=verbose))
        else:
            history = self._model.evaluate(train_x,train_y,
                                           batch_size=self._batch_size,
                                           verbose=verbose)
        history=np.transpose(history)
        self._result = dict(zip(self._model.metrics_names, history))
        