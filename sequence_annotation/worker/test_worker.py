"""This submodule provides trainer to train model"""
import tensorflow as tf
import keras.backend as K
import json
import numpy as np
import pandas as pd
from . import DataGenerator
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
        root_path = './'+self._path_root
        self._create_folder(["test"])
    def after_work(self):
        data = json.loads(pd.Series(self._result).to_json(orient='index'))
        with open(self._path_root + '/test/evaluate.json', 'w') as outfile:  
            json.dump(data, outfile,indent=4)
    def _prepare_data(self,x_data,y_data,batch_size):
        generator = DataGenerator(x_data,y_data,batch_size)
        return generator
    def _evaluate_by_generator(self):
        test_x = self._data['testing']['inputs']
        test_y = self._data['testing']['answers']
        test_data = self._prepare_data(test_x,test_y,self._batch_size)
        history = self._model.evaluate_generator(test_data,
                                                 verbose=int(self.is_verbose_visible))
        return history
    def _evaluate_by_fit(self):
        test_x = self._data['testing']['inputs']
        test_y = self._data['testing']['answers']          
        history = self._model.evaluate(x=test_x,y=test_y,
                                       verbose=int(self.is_verbose_visible),
                                       batch_size=self._batch_size)
        return history
    def work(self):
        test_x = self._data['testing']['inputs']
        test_y = self._data['testing']['answers']
        history = []
        if self._use_generator:
            history.append(self._evaluate_by_generator())
        else:
            if sample_weight_setting is not None:
                raise Exception("Only generator can use sample weight")
            history = self._evaluate_by_fit()
        history=np.transpose(history)
        self._result = dict(zip(self._model.metrics_names, history))
        