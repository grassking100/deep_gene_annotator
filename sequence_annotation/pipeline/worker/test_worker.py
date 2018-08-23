"""This submodule provides trainer to train model"""
import tensorflow as tf
import keras.backend as K
import json
import numpy as np
import pandas as pd
from ...model.data_generator import DataGenerator
from .worker import Worker
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

class TestWorker(Worker):
    def __init__(self,batch_size=32,
                 use_generator=False,
                 test_per_batch=False,
                 path_root=None,
                 is_verbose_visible=True):
        super().__init__(path_root)
        self._result = {}
        self._batch_size = batch_size
        self._use_generator = use_generator
        self._test_per_batch = test_per_batch
        self.is_verbose_visible = is_verbose_visible
    def before_work(self):
        if self._path_root is not None:
            self._create_folder(["test"])
    def after_work(self):
        if self._path_root is not None:
            data = json.loads(pd.Series(self._result).to_json(orient='index'))
            with open(self._path_root + '/test/evaluate.json', 'w') as outfile:  
                json.dump(data, outfile,indent=4)
    def _prepare_data(self,x_data,y_data,batch_size):
        generator = DataGenerator(x_data,y_data,batch_size)
        return generator
    def _get_data(self):
        test_x = self.data['testing']['inputs']
        test_y = self.data['testing']['answers']
        if self._test_per_batch:  
            test_x_list = np.split(test_x,len(test_x))
            test_y_list = np.split(test_y,len(test_x))
        else:
            test_x_list = [test_x]
            test_y_list = [test_y]
        return (np.array(test_x_list),np.array(test_y_list))
    def _evaluate_by_generator(self):
        test_x_list,test_y_list = self._get_data()
        history = []
        for test_x,test_y in zip(test_x_list,test_y_list):
            test_data = self._prepare_data(test_x,test_y,self._batch_size)
            temp = self.model.evaluate_generator(test_data,verbose=int(self.is_verbose_visible))
            history.append(temp)
        return history
    def _evaluate_by_fit(self):
        test_x_list,test_y_list = self._get_data()
        history = []
        for test_x,test_y in zip(test_x_list,test_y_list):
            temp = self.model.evaluate(x=test_x,y=test_y,
                                        verbose=int(self.is_verbose_visible),
                                        batch_size=self._batch_size)
            history.append(temp)
        return history
    def _validate(self):
        """Validate required data"""
        if self.model is None:
            raise Exception("Model must be passed to worker")
        if self.data is None:
            raise Exception("Data must be passed to worker")
    def work(self):
        self._validate()
        if self._use_generator:
            history = self._evaluate_by_generator()
        else:
            history = self._evaluate_by_fit()
        history=np.transpose(history)
        self._result = dict(zip(self.model.metrics_names, history))