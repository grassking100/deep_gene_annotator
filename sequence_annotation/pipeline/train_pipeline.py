"""This submodule provides class to defined training pipeline"""
import os
import json
import pandas as pd
from time import strftime, gmtime
from os.path import expanduser
from .basic_pipeline import BasicPipeline
from .worker.train_worker import TrainWorker
class TrainPipeline(BasicPipeline):
    """a pipeline about training model"""
    def _get_class_count(self):
        return self._preprocessed_data['training']['annotation_count']
    @property
    def _setting_saved_name(self):
        return '/train_setting.json'
    def _load_previous_result(self):
        path = self._work_setting['previous_result_file']
        result = pd.read_csv(path,index_col=False, skiprows=None).to_dict(orient='list')
        return result
    def _before_execute(self):
        super()._before_execute()
        if self._work_setting['initial_epoch']==0:
            mode_id=self._work_setting['mode_id']
            path_root = self._work_setting['path_root']+"/"+self._id+"/"+mode_id
            if self._is_prompt_visible:
                print('Create record folder:'+path_root)
            self._create_folder(path_root)
        else:
            if self._is_prompt_visible:
                print('Loading previous result')
            self._worker.result = self._load_previous_result()
    def _init_worker(self):
        setting = self._work_setting
        mode_id=setting['mode_id']
        path_root=setting['path_root']+"/"+str(self._id)+"/"+mode_id
        self._worker=TrainWorker(path_root,setting['epoch'],
                                 setting['batch_size'],
                                 self._model,
                                 self._processed_data,
                                 initial_epoch=setting['initial_epoch'],
                                 period=setting['period'],
                                 validation_split=setting['validation_split'],
                                 use_generator=setting['use_generator'])
        self._worker.is_verbose_visible=self._is_prompt_visible
        self._worker.is_prompt_visible=self._is_prompt_visible
    def _setting_to_saved(self):
        saved = super()._setting_to_saved()
        training = self._preprocessed_data['training']
        if 'annotation_count' in training.keys():
            saved['annotation_count'] = training['annotation_count']
        return saved