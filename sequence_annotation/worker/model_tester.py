"""This submodule provides trainer to train model"""
import tensorflow as tf
import keras.backend as K
from . import ModelWorker
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
class ModelTester(ModelWorker):
    def predict(self, data):
        """Test model"""
        self._validate(data)
        return self.model.predict(data)
    def evaluate(self, x, y):
        """Test model"""
        self._validate(x,y)
        return self.model.evaluate(x, y)
        