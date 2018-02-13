"""This submodule provides class to defined Pipeline"""
import os
import errno
from abc import ABCMeta, abstractmethod
from time import gmtime, strftime
from keras.models import load_model
from keras.utils import plot_model
import pandas as pd
from . import CustomObjectsFacade
from . import ModelFacade
from . import TrainDataLoader
from . import TrainSettingParser
from . import ModelSettingParser
from . import handle_alignment_files

class Pipeline(metaclass=ABCMeta):
    def __init__(self, setting_file, worker):
        self._setting_file = setting_file
        self._worker = worker
        self._setting = None
        self._weights = None
        self._model = None
    @abstractmethod
    def _validate_required(self):
        pass
    def _validate_setting(self):
        if self._setting is None:
            raise Exception("Setting is not be set properly")
    @abstractmethod
    def _parse(self):
        pass
    @abstractmethod
    def _load_data(self):
        pass
    @abstractmethod
    def _init_worker(self):
        pass
    @abstractmethod
    def execute(self):
        pass
    def _init_model(self):
        model_facade = ModelFacade(self._setting,self._weights)
        self._model = model_facade.model()
    def _load_model(self):
        previous_status_root = self._setting['previous_status_root']
        facade = CustomObjectsFacade(self._setting['ANN_TYPES'],
                                     self._setting['output_dim'],
                                     self._setting['terminal_signal'],
                                     self._weights,
                                     'accuracy','loss')
        self._model = load_model(previous_status_root+'.h5', facade.custom_objects)
    def _calculate_weights(self,count):
        self._weights = []
        weights = []
        total_count = 0
        for key in self._setting['ANN_TYPES']:
            weights.append(1/count[key])
            total_count += count[key]
        sum_weight = sum(weights)
        self._weights = [total_count*weight/sum_weight for weight in weights]
    def _create_folder(self,path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError as erro:
            if erro.errno != errno.EEXIST:
                raise
class TrainPipeline(Pipeline):
    def __init__(self, setting_file, worker):
        super().__init__(setting_file, worker)
        self.__folder_name = None
        self._parse()
        self._init_folder_name()
        if self._setting['is_prompt_visible']:
            print("Loading data")
        self._load_data()
        if self._setting['previous_epoch'] > 0:
            if self._setting['is_prompt_visible']:
                print("Load previous model")
            self._load_model()
        else:
            if self._setting['is_prompt_visible']:
                print("Initialize the model")
            self._init_model()
        if self._setting['is_prompt_visible']:
            print("Initialize training worker")
        self._init_worker()
    def _init_folder_name(self):
        self.__folder_name = (self._setting['outputfile_root']+'/'+
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
        (train_x,
         train_y,
         train_x_count) = handle_alignment_files(self._setting['training_files'],
                                                 self._setting['training_answers'],
                                                 self._setting['ANN_TYPES'])
        (val_x, val_y) = handle_alignment_files(self._setting['validation_files'],
                                                self._setting['validation_answers'],
                                                self._setting['ANN_TYPES'])[0:2]
        if self._setting['use_weights']:
            self._calculate_weights(train_x_count)
        loader = TrainDataLoader()
        loader.load(self._worker, train_x, train_y,
                    val_x , val_y,self._setting['terminal_signal'])
    def _init_worker(self):
        self._worker.model = self._model
    def _validate_required(self):
        keys = ['_setting','_model']
        for key in keys:
            if getattr(self,key) is None:
                raise Exception("TrainPipeline needs "+key+" to complete the quest")
    def __get_whole_file_path(self,progress_number):
        file_name = (self._setting['train_id']+'_'+self._setting['mode_id']+
                     '_progress_'+str(progress_number)+'_')
        date = strftime("%Y_%b_%d", gmtime())
        whole_file_path=self.__folder_name+"/"+file_name+date
        return whole_file_path
    def __save_setting(self):
        model_path=self.__get_whole_file_path(0)
        self._model.save(model_path+'.h5')
        plot_model(self._model, show_shapes=True,
                   to_file=self.__folder_name+"/"+self._setting['model_image_name'])
        data = dict(self._setting)
        data['folder_name'] = str(self.__folder_name)
        data['save_time'] = strftime("%Y_%b_%d", gmtime())
        for type_, value in zip(self._setting['ANN_TYPES'],self._weights):
            data[type_+'_weight'] = value
        df = pd.DataFrame(list(data.items()),columns=['attribute','value'])
        df.to_csv(self.__folder_name+"/"+self._setting['setting_record_name'],index =False)
    def __load_previous_result(self):
        result = pd.read_csv(self._setting['previous_status_root']+".csv",
                             index_col=False, skiprows=None).to_dict(orient='list')
        return result
    def execute(self):
        self._validate_required()
        if self._setting['previous_epoch']==0:
            if self._setting['is_prompt_visible']:
                print('Create record file:'+self.__folder_name+"/"+self._setting['setting_record_name'])
            self._create_folder(self.__folder_name)
            self.__save_setting()
        else:
            if self._setting['is_prompt_visible']:
                print('Loading previous result')
            self._worker.result = self.__load_previous_result()
        for progress in range(self._setting['previous_epoch'],
                              self._setting['progress_target'],self._setting['step']):
            if progress+self._setting['step'] > self._setting['progress_target']:
                step = self._setting['progress_target'] - progress
                finished_progress_number = self._setting['progress_target']
            else:
                step = self._setting['step']
                finished_progress_number = progress + self._setting['step']
            if self._setting['is_prompt_visible']:
                print("\n"+str(progress)+"/"+str(self._setting['progress_target']))
            path = self.__get_whole_file_path(finished_progress_number)
            self._worker.train(step, self._setting['batch_size'],
                                  True, int(self._setting['is_verbose_visible']),path+'/log/')
            df = pd.DataFrame().from_dict(self._worker.result)
            if self._setting['is_prompt_visible']:
                print("Save metrics result to "+path+'.csv')
                print("Save model result to "+path+'.h5')
            df.to_csv(path+'.csv',index =False)
            self._model.save(path+'.h5')
        if self._setting['is_prompt_visible']:
            print('End of training')