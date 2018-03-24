"""This submodule provides class to defined training pipeline"""
import pandas as pd
from . import SeqAnnDataHandler
from . import SimpleDataHandler
from . import Pipeline
from . import ModelTrainer
class TrainPipeline(Pipeline):
    def _prepare_for_compile(self):
        weight_setting = self._work_setting['weight_setting']
        if weight_setting['use_weights']:
            class_counts = self._preprocessed_data['training']['annotation_count']
            self._weighted=self._model_handler.get_weights(class_counts=class_counts,
                                                           method_name=weight_setting['method'])
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
class TrainSeqAnnPipeline(TrainPipeline):
    def __init__(self,id_,work_setting_path,model_setting_path,is_prompt_visible=True):
        super().__init__(id_, work_setting_path,model_setting_path,is_prompt_visible)
        self._data_handler = SeqAnnDataHandler
        self._worker = ModelTrainer()
    def _load_data(self):
        self._preprocessed_data = {}
        data_path = self._work_setting['data_path']
        ann_types=self._model_setting['annotation_types']
        for name,path in data_path.items():
            temp = self._data_handler.get_data(path['inputs'],
                                               path['answers'],
                                               ann_types=ann_types,
                                               discard_invalid_seq=True)
            self._preprocessed_data[name] = temp
    def _setting_to_saved(self):
        saved = super()._setting_to_saved()
        saved['annotation_count'] = self._preprocessed_data['training']['annotation_count']
        return saved
class TrainSimplePipeline(TrainPipeline):
    def __init__(self,id_,work_setting_path,model_setting_path,is_prompt_visible=True):
        super().__init__(id_, work_setting_path,model_setting_path,is_prompt_visible)
        self._data_handler = SimpleDataHandler
        self._worker = ModelTrainer()
    def _load_data(self):
        self._preprocessed_data = {}
        data_path = self._work_setting['data_path']
        for name,path in data_path.items():
            temp = self._data_handler.get_data(path['inputs'],path['answers'])
            self._preprocessed_data[name] = temp