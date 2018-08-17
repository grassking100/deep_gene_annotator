import os
import sys
from os.path import abspath, expanduser
sys.path.append(abspath(expanduser(__file__+"/../..")))
import unittest
import shutil
import pandas as pd
from . import PipelineFactory
from . import read_json
from . import BatchPipeline
class TestBatchPipeline(unittest.TestCase):
    def test_runable(self):
        user_setting = {}
        work_setting_path=abspath(expanduser(__file__+'/../setting/batch_train_setting.json'))
        model_setting_path=abspath(expanduser(__file__+'/../setting/model_setting.json'))
        train_id='test_runable'
        try:
            pipeline = BatchPipeline(data_type='sequence_annotation',is_prompt_visible=False)
            work_setting = read_json(work_setting_path)
            model_setting = read_json(model_setting_path)
            pipeline.execute(train_id,work_setting,model_setting)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
        finally:
            path = abspath(expanduser(__file__+'/../batch_result/'+train_id))
            if os.path.exists(path):
                shutil.rmtree(path)
