"""This submodule provides trainer to train model"""
import tensorflow as tf
import keras.backend as K
import json
import pandas as pd
from ...process.worker import Worker
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

class TestWorker(Worker):
    def __init__(self,wrapper,path_root=None,is_verbose_visible=True):
        super().__init__(wrapper,path_root)
        self._result = {}
        self.is_verbose_visible = is_verbose_visible
        self.evaluate_generator_wrapper = None

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
        history = self.wrapper(self.model,self.data)
        try:
            iter(history)
        except TypeError:
            history = [history]
        self._result = dict(zip(self.model.metrics_names, history))
        
    def _validate(self):
        super()._validate()
