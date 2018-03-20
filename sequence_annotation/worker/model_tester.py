"""This submodule provides trainer to train model"""
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import numpy as np
import os
from . import DataGenerator
from . import ModelWorker
from . import DataValidator, DictValidator, AttrValidator
from . import ResultHistory
class ModelTester(ModelWorker):
    def predict(self, data):
        """Test model"""
        self._validate(data)
        return self.model.predict(data)
    def evaluate(self, x, y):
        """Test model"""
        self._validate(data)
        return self.model.evaluate(x, y)
        