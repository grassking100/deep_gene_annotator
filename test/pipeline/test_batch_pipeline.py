import os
import sys
from os.path import abspath, expanduser
sys.path.append(abspath(expanduser(__file__+"/../..")))
import unittest
import shutil
import pandas as pd
import argparse
from time import strftime, gmtime, time
from . import PipelineFactory
from . import JsonReader
from . import BatchPipeline
class TestBatchPipeline(unittest.TestCase):
    def test_runable(self):
        user_setting = {}
        work_setting_path=abspath(expanduser(__file__+'/../setting/batch_train_setting.json'))
        model_setting_path=abspath(expanduser(__file__+'/../setting/model_setting.json'))
        train_id='test_runable'
        try:
            pipeline = BatchPipeline(data_type='sequence_annotation')
            work_setting = JsonReader().read(work_setting_path)
            model_setting = JsonReader().read(model_setting_path)
            pipeline.execute(train_id,work_setting,model_setting)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            path = abspath(expanduser(__file__+'/../batch_result/'+train_id))
            print(path)
            if os.path.exists(path):
                shutil.rmtree(path)
if __name__=="__main__":
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestBatchPipeline)
    unittest.main()