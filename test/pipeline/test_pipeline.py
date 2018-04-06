import unittest
import os
from os.path import abspath, expanduser
import sys
import shutil
import pandas as pd
import argparse
sys.path.append(abspath(expanduser(__file__+"/../..")))
from sequence_annotation.pipeline.train_pipeline import TrainSeqAnnPipeline
from sequence_annotation.pipeline.test_pipeline import TestSeqAnnPipeline
class TestPipeline(unittest.TestCase):
    def test_runable(self):
        user_setting = {}
        user_setting['work_setting_path']=__file__+'/../setting/train_setting.json'
        user_setting['model_setting_path']=__file__+'/../setting/model_setting.json'
        user_setting['train_id']='test_runable'
        try:
            train_pipeline = TrainSeqAnnPipeline(user_setting['train_id'],
                                                 user_setting['work_setting_path'],
                                                 user_setting['model_setting_path'],False)
            train_pipeline.execute()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            path = abspath(expanduser(__file__+'/../result/'+user_setting['train_id']))
            if os.path.exists(path):
                shutil.rmtree(path)
    def test_constant(self):
        user_setting = {}
        user_setting['work_setting_path']=__file__+'/../setting/train_setting.json'
        user_setting['model_setting_path']=__file__+'/../setting/model_setting.json'
        user_setting['train_id']='test_constant'
        try:
            train_pipeline = TrainSeqAnnPipeline(user_setting['train_id'],
                                                 user_setting['work_setting_path'],
                                                 user_setting['model_setting_path'],False)
            train_pipeline.execute()
            path =  abspath(expanduser(__file__+'/../result/'+user_setting['train_id']+'/train/result/epoch_03.csv'))
            data = pd.read_csv(path)
            self.assertEqual((data['constant_layer']==300).tolist(),[True]*3)
            self.assertEqual((data['val_constant_layer']==300).tolist(),[True]*3)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            path = abspath(expanduser(__file__+'/../result/'+user_setting['train_id']))
            if os.path.exists(path):
                shutil.rmtree(path)
                
    def test_train_test(self):
        user_setting = {}
        user_setting['train_setting_path']=__file__+'/../setting/train_setting.json'
        user_setting['test_setting_path']=__file__+'/../setting/test_setting.json'
        user_setting['model_setting_path']=__file__+'/../setting/model_setting.json'
        user_setting['id']='test_train_test'
        try:
            train_pipeline = TrainSeqAnnPipeline(user_setting['id'],
                                                 user_setting['train_setting_path'],
                                                 user_setting['model_setting_path'],False)
            train_pipeline.execute()
            
            test_pipeline = TestSeqAnnPipeline(user_setting['id'],
                                               user_setting['test_setting_path'],
                                               user_setting['model_setting_path'],False)
            test_pipeline.execute()
            self.assertEqual(dict(test_pipeline.worker.result)['loss'] <= 0.05,True)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            path =  abspath(expanduser(__file__+'/../result/'+user_setting['id']))               
            if os.path.exists(path):
                shutil.rmtree(path)
if __name__=="__main__":
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestPipeline)
    unittest.main()
