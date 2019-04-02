"""This submodule provides trainer to train model"""
import warnings
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
from ...process.data_generator import DataGenerator
from ..function.work_generator import EvaluateGenerator


class TestWorker(Worker):
    def __init__(self,test_generator=None,evaluate_generator=None):
        super().__init__()
        self._evaluate_generator = evaluate_generator or EvaluateGenerator()
        self._test_generator = test_generator or DataGenerator()
        
    def before_work(self,path=None):
        if path is not None:
            create_folder("./"+path+"/test")
    def after_work(self,path=None):
        warnings.warn("This method is deprecated,it will be replaced by using callback in Keras",
                      DeprecationWarning)
        if path is not None:
            data = json.loads(pd.Series(self._result).to_json(orient='index'))
            with open(path + '/test/evaluate.json', 'w') as outfile:  
                json.dump(data, outfile,indent=4)

    def work(self):
        data = self.data['testing']
        self._validate()
        self._test_generator.x_data=data['inputs']
        self._test_generator.y_data=data['answers']
        self._evaluate_generator.generator = self._test_generator
        self._evaluate_generator.model = self.model
        history = self._evaluate_generator()
        try:
            iter(history)
        except TypeError:
            history = [history]
        self.result = dict(zip(self.model.metrics_names, history))
