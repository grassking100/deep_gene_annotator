import unittest
import os
import sys
import shutil
import pandas as pd
import argparse
sys.path.append(os.path.abspath("~/../../.."))
from sequence_annotation.pipeline.train_pipeline import TrainSeqAnnPipeline
from sequence_annotation.worker.model_trainer import ModelTrainer
class TestPipeline(unittest.TestCase):
    def test_runable(self):
        user_setting = {}
        user_setting['work_setting_path']='~/deep_learning/sequence_annotation/test/pipeline/setting/training_setting.json'
        user_setting['model_setting_path']='~/deep_learning/sequence_annotation/test/pipeline/setting/model_setting.json'
        user_setting['train_id']='test_runable'
        try:
            train_pipeline = TrainSeqAnnPipeline(user_setting['train_id'],user_setting['work_setting_path'],
                                                 user_setting['model_setting_path'],False)
            train_pipeline.execute()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            if os.path.exists('result/'+user_setting['train_id']):
                shutil.rmtree('result/'+user_setting['train_id'])
    def test_constant(self):
        user_setting = {}
        user_setting['work_setting_path']='~/deep_learning/sequence_annotation/test/pipeline/setting/training_setting.json'
        user_setting['model_setting_path']='~/deep_learning/sequence_annotation/test/pipeline/setting/model_setting.json'
        user_setting['train_id']='test_constant'
        try:
            train_pipeline = TrainSeqAnnPipeline(user_setting['train_id'],user_setting['work_setting_path'],
                                                 user_setting['model_setting_path'],False)
            train_pipeline.execute()
            data = pd.read_csv('result/'+user_setting['train_id']+'/test/result/epoch_3.csv')
            self.assertEqual((data['constant_layer']==300).tolist(),[True]*3)
            self.assertEqual((data['val_constant_layer']==300).tolist(),[True]*3)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            if os.path.exists('result/'+user_setting['train_id']):
                shutil.rmtree('result/'+user_setting['train_id'])
if __name__=="__main__":
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestPipeline)
    unittest.main()
