import unittest
from sequence_annotation.process.model_processor import SimpleModel
from sequence_annotation.keras.process.model_creator import ModelCreator
from keras.models import Sequential
from keras.layers import Dense, Activation
import json
from os.path import abspath, expanduser

class TestModelProcessor(unittest.TestCase):
    def test_simple_model(self):
        try:
            model = Sequential([
                Dense(32, input_shape=(2,)),
                Activation('relu'),
                Dense(2),
                Activation('softmax')
            ])
            simple_model = SimpleModel(model)
            simple_model.before_process()
            simple_model.process()
            simple_model.after_process()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")

    def test_model_builder(self):
        try:
            model_setting_path = abspath(expanduser(__file__+'/../../setting/model_setting.json'))
            with open(model_setting_path) as fp:
                model_setting=json.load(fp)
            model = ModelCreator(model_setting)
            model.before_process()
            model.process()
            model.after_process()        
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")