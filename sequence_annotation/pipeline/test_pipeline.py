"""This submodule provides class to defined test pipeline"""
import pandas as pd
from os.path import expanduser
from . import SeqAnnDataHandler
from . import SimpleDataHandler
from . import Pipeline
from . import TestWorker

class TestPipeline(Pipeline):
    def _init_worker(self):
        setting = self._work_setting
        mode_id=setting['mode_id']
        path_root=setting['path_root']+"/"+str(self._id)+"/"+mode_id
        self._worker=TestWorker(path_root,setting['batch_size'],
                                self._model,self._processed_data,
                                use_generator=setting['use_generator'])
        self._worker.is_verbose_visible=self._is_prompt_visible
        self._worker.is_prompt_visible=self._is_prompt_visible
    def _prepare_for_compile(self):
        weight_setting = self._work_setting['weight_setting']
        if weight_setting['use_weights']:
            class_counts = self._preprocessed_data['testing']['annotation_count']
            self._weighted=self._model_handler.get_weights(class_counts=class_counts,
                                                           method_name=weight_setting['method'])

class TestSeqAnnPipeline(TestPipeline):
    def __init__(self,id_,work_setting_path,model_setting_path,is_prompt_visible=True):
        super().__init__(id_, work_setting_path,model_setting_path,is_prompt_visible)
        self._data_handler = SeqAnnDataHandler