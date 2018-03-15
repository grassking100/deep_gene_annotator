from . import ModelBuildFacade
from . import TrainDataLoader
from . import TrainSettingParser
from . import ModelSettingParser
from . import handle_alignment_files
from . import Pipeline
import pandas as pd
from time import gmtime, strftime, time
import numpy as np
class TrainPipeline(Pipeline):
    def __init__(self, setting_file, worker):
        super().__init__(setting_file, worker)
        self._weights = None
        self._counts = None
        self._folder_name = None
        self._parse()
        self._init_folder_name()
        if self._setting['is_prompt_visible']:
            print("Loading data")
        self._load_data()
        self._init_custom_objects()
        if self._setting['previous_epoch'] > 0:
            if self._setting['is_prompt_visible']:
                print("Load previous model")
            self._load_model()
        else:
            if self._setting['is_prompt_visible']:
                print("Initialize the model")
            self._init_model()
        if self._setting['is_prompt_visible']:
            print("Compile the model")
        self._compile_model()
        if self._setting['is_prompt_visible']:
            print("Initialize training worker")
        self._init_worker()
    def _init_folder_name(self):
        self._folder_name = (self._setting['outputfile_root']+'/'+
                              self._setting['train_id']+'/'+self._setting['mode_id'])
    def _parse(self):
        train_parser = TrainSettingParser(self._setting_file['training_setting_path'])
        model_parser = ModelSettingParser(self._setting_file['model_setting_path'])
        model_parser.parse()
        train_parser.parse()
        self._setting = dict(train_parser.setting)
        self._setting['train_id'] = self._setting_file['train_id']
        self._setting.update(model_parser.setting)
    def _load_data(self):
        (train_x,train_y,
         self._counts) = handle_alignment_files(self._setting['training_files'],
                                                self._setting['training_answers'])
        (val_x, val_y) = handle_alignment_files(self._setting['validation_files'],
                                                self._setting['validation_answers'])[0:2]
        if self._setting['use_weights']:
            self._weights = np.multiply(self._calculate_weights(self._counts),len(self._setting['ANN_TYPES']))
        loader = TrainDataLoader()
        loader.load(self._worker, train_x, train_y,
                    val_x , val_y,self._setting['terminal_signal'])
    def _init_worker(self):
        self._worker.model = self._model
        self._worker.settings = self._get_setting()
    def _validate_required(self):
        keys = ['_setting','_model']
        for key in keys:
            if getattr(self,key) is None:
                raise Exception("TrainPipeline needs "+key+" to complete the quest")
    def _save_setting(self):
        data = dict(self._setting)
        data['folder_name'] = str(self._folder_name)
        data['save_time'] = strftime("%Y_%b_%d", gmtime())
        for type_, value in zip(self._setting['ANN_TYPES'],self._weights):
            data[type_+'_weight'] = value
        for type_,number in self._counts.items():
            data[type_+'_number'] = number
        df = pd.DataFrame(list(data.items()),columns=['attribute','value'])
        df.to_csv(self._folder_name+"/"+self._setting['setting_record_name'],index=False)
    def _load_previous_result(self):
        path = self._setting['previous_result_file']
        result = pd.read_csv(path,index_col=False, skiprows=None).to_dict(orient='list')
        return result
    def _calculate_weights(self,count):
        weights = []
        total_count = 0
        for key in self._setting['ANN_TYPES']:
            weights.append(1/count[key])
            total_count += count[key]
        sum_weight = sum(weights)
        return [weight/sum_weight for weight in weights]
    def _get_setting(self):
        setting = {}
        setting['initial_epoch'] = self._setting['previous_epoch']
        setting['batch_size'] = self._setting['batch_size']
        setting['shuffle'] = True
        setting['epochs'] = self._setting['progress_target']
        setting['is_prompt_visible'] = self._setting['is_prompt_visible']
        setting['is_verbose_visible'] = self._setting['is_verbose_visible']
        setting['period'] = self._setting['step']
        setting['file_path_root'] = self._folder_name
        setting['weights'] = self._weights
        setting['model_image_name'] = self._setting['model_image_name']
        return setting
    def execute(self):
        self._validate_required()
        if self._setting['previous_epoch']==0:
            if self._setting['is_prompt_visible']:
                print('Create record folder:'+self._folder_name+"/"+self._setting['setting_record_name'])
            self._create_folder(self._folder_name)
            self._save_setting()
        else:
            if self._setting['is_prompt_visible']:
                print('Loading previous result')
            self._worker.result = self._load_previous_result()
        if self._setting['is_prompt_visible']:
            print('Start training('+strftime("%Y-%m-%d %H:%M:%S",gmtime())+")")
        start_time = time()
        self._worker.train()
        end_time = time()
        time_spend = end_time - start_time
        if self._setting['is_prompt_visible']:
            print('End training('+strftime("%Y-%m-%d %H:%M:%S",gmtime())+")")
            print("Spend time: "+strftime("%H:%M:%S", gmtime(time_spend)))