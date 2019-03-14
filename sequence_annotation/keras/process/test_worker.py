"""This submodule provides trainer to train model"""
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
import json
import pandas as pd
from ...process.worker import Worker
from ...function.data_generator import DataGenerator

class TestWorker(Worker):
    def __init__(self,path_root=None,*args,**kwargs):
        super().__init__(path_root)
        self._args = args
        self._kwargs = kwargs

    def before_work(self):
        if self._path_root is not None:
            create_folder("./"+path+"/test")
    def after_work(self):
        if self._path_root is not None:
            data = json.loads(pd.Series(self._result).to_json(orient='index'))
            with open(self._path_root + '/test/evaluate.json', 'w') as outfile:  
                json.dump(data, outfile,indent=4)
    def work(self):
        self._validate()
        item = self.data['testing']
        if 'batch_size' in self._kwargs:
            batch_size = self._kwargs['batch_size']
        else:
            batch_size = 32
        test_data = DataGenerator(item['inputs'],item['answers'],batch_size=batch_size)
        if 'batch_size' in self._kwargs:
            del self._kwargs['batch_size']
        history = self.model.evaluate_generator(test_data,*self._args,**self._kwargs)
        try:
            iter(history)
        except TypeError:
            history = [history]
        self._result = dict(zip(self.model.metrics_names, history))

