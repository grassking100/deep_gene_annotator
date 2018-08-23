"""This submodule provides class to defined test pipeline"""
import pandas as pd
import os
import json
from time import gmtime, strftime
from os.path import expanduser
from .basic_pipeline import BasicPipeline
from .worker.test_worker import TestWorker

class TestPipeline(BasicPipeline):
    def _get_class_count(self):
        return self._preprocessed_data['testing']['annotation_count']
    @property
    def _setting_saved_name(self):
        return '/test_setting.json'
    def _init_worker(self):
        setting = self._work_setting
        mode_id=setting['mode_id']
        path_root=setting['path_root']+"/"+str(self._id)+"/"+mode_id
        self._worker=TestWorker(path_root,setting['batch_size'],
                                self._model,self._processed_data,
                                use_generator=setting['use_generator'],
                                test_per_batch=setting['test_per_batch'])
        self._worker.is_verbose_visible=self._is_prompt_visible
        self._worker.is_prompt_visible=self._is_prompt_visible