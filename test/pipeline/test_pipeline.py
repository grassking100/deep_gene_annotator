import os
import sys
from os.path import abspath, expanduser
sys.path.append(abspath(expanduser(__file__+"/../..")))
import unittest
import shutil
import pandas as pd
from . import PipelineFactory
from . import read_json

class TestPipeline(unittest.TestCase):
    def test_runable(self):
        user_setting = {}
        work_setting_path=abspath(expanduser(__file__+'/../setting/train_setting.json'))
        model_setting_path=abspath(expanduser(__file__+'/../setting/model_setting.json'))
        train_id='test_runable'
        try:
            train_pipeline = PipelineFactory().create('train','sequence_annotation',False)
            work_setting = read_json(work_setting_path)
            model_setting = read_json(model_setting_path)
            train_pipeline.execute(train_id,work_setting,model_setting)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            path = abspath(expanduser(__file__+'/../result/'+train_id))
            if os.path.exists(path):
                shutil.rmtree(path)
    def test_constant_metric(self):
        user_setting = {}
        work_setting_path = abspath(expanduser(__file__+'/../setting/train_setting.json'))
        model_setting_path = abspath(expanduser(__file__+'/../setting/model_setting.json'))
        train_id='test_constant'
        try:
            train_pipeline = PipelineFactory().create('train','sequence_annotation',False)
            work_setting = read_json(work_setting_path)
            model_setting = read_json(model_setting_path)
            train_pipeline.execute(train_id,work_setting,model_setting)
            path =  abspath(expanduser(__file__+'/../result/'+train_id+'/train/result/epoch_03.csv'))
            data = pd.read_csv(path)
            self.assertEqual((data['batch_counter_layer']==3).tolist(),[True]*3)
            self.assertEqual((data['val_batch_counter_layer']==3).tolist(),[True]*3)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            path = abspath(expanduser(__file__+'/../result/'+train_id))
            if os.path.exists(path):
                shutil.rmtree(path)
           
    def test_train_test(self):
        user_setting = {}
        train_setting_path = abspath(expanduser(__file__+'/../setting/train_setting.json'))
        test_setting_path = abspath(expanduser(__file__+'/../setting/test_setting.json'))
        model_setting_path = abspath(expanduser(__file__+'/../setting/model_setting.json'))
        id_='test_train_test'
        try:
            train_pipeline = PipelineFactory().create('train','sequence_annotation',False)
            train_setting = read_json(train_setting_path)
            test_setting = read_json(test_setting_path)
            model_setting = read_json(model_setting_path)
            train_pipeline.execute(id_,train_setting,model_setting)
            test_pipeline = PipelineFactory().create('test','sequence_annotation',False)
            test_pipeline.execute(id_,test_setting,model_setting)
            self.assertEqual(dict(test_pipeline.worker.result)['loss'] <= 0.05,True)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            path = abspath(expanduser(__file__+'/../result/'+id_))
            if os.path.exists(path):
                shutil.rmtree(path)
