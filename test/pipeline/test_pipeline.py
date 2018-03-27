import unittest
import os
import sys
import shutil
sys.path.append(os.path.abspath("~/../../.."))
import argparse
from sequence_annotation.pipeline.train_pipeline import TrainSeqAnnPipeline
from sequence_annotation.worker.model_trainer import ModelTrainer
class TestPipeline(unittest.TestCase):
    def test_runable(self):
        user_setting = {}
        user_setting['work_setting_path']='~/deep_learning/sequence_annotation/test/pipeline/setting/training_setting.json'
        user_setting['model_setting_path']='~/deep_learning/sequence_annotation/test/pipeline/setting/model_setting.json'
        user_setting['train_id']='simple_run'
        try:
            train_pipeline = TrainSeqAnnPipeline("pipeline_test",user_setting['work_setting_path'],
                                                 user_setting['model_setting_path'],False)
            train_pipeline.execute()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            if os.path.exists('result'):
                shutil.rmtree('result')
if __name__=="__main__":
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestPipeline)
    unittest.main()
