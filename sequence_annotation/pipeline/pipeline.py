"""This submodule provides class to defined Pipeline"""
import os
import errno
from abc import ABCMeta, abstractmethod
from time import strftime, gmtime, time
from shutil import copyfile
from . import SettingParser
from . import ModelHandler
import json
class Pipeline(metaclass=ABCMeta):
    def __init__(self, id_, work_setting_path,model_setting_path,is_prompt_visible=True):
        self._id= id_
        self._is_prompt_visible = is_prompt_visible
        self._work_setting_path = os.path.expanduser(work_setting_path)
        self._model_setting_path = os.path.expanduser(model_setting_path)
        self._data_handler= None
        self._worker = None
        self._work_setting_parser = SettingParser()
        self._model_setting_parser = SettingParser()
        self._model_handler = ModelHandler()
        self._work_setting = None
        self._model_setting = None
        self._preprocessed_data = None
        self._processed_data = None
        self._weighted = None
        self._model = None
    def print_prompt(self,value):
        if self._is_prompt_visible:
            print(value)
    def execute(self):
        self.print_prompt("Parsing setting...")
        self._parse_setting()
        self.print_prompt("Initializing model..")
        self._init_model()
        self.print_prompt("Loading data...")
        self._load_data()
        self.print_prompt("Processing data...")
        self._process_data()
        self.print_prompt("Prepare for compiling ...")
        self._prepare_for_compile()
        self.print_prompt("Compiling model...")
        self._compile_model()
        self.print_prompt("Initializing worker...")
        self._init_worker()
        self._before_execute()
        self.print_prompt("Executing...")
        self._execute()
        self._after_execute()
    @abstractmethod
    def _prepare_for_compile(self):
        pass
    def _parse_setting(self):
        self._work_setting = self._work_setting_parser.parse(self._work_setting_path)
        self._model_setting = self._model_setting_parser.parse(self._model_setting_path)
    def _init_model(self):
        build_model = True
        if 'initial_epoch' in self._work_setting.keys():
            if int(self._work_setting['initial_epoch']) > 0:
                build_model = False
        if not build_model: 
            self.print_prompt("\tLoading model...")
            model_path = self._work_setting['trained_model_path']
            self._model = self._model_handler.load_model(model_path)
            self._model.load_weights(model_path, by_name=True)
        else:
            self.print_prompt("\tBuilding model...")
            self._model = self._model_handler.build_model(self._model_setting)

    def _padding(self, data_dict, value):
        padded = {}
        for data_kind, data in data_dict.items():
            inputs = data['inputs']
            answers = data['answers']
            inputs, answers = self._data_handler.padding(inputs,answers,value)
            padded[data_kind]= {"inputs":inputs,"answers":answers}
        return padded
    def _process_data(self):
        self._processed_data = {}
        for data_kind,data in self._preprocessed_data.items():
            inputs,answers = self._data_handler.to_vecs(data['data_pair'].values())
            self._processed_data[data_kind] = {"inputs":inputs,"answers":answers}
        for preprocess in self._work_setting['preprocess']:
            if preprocess['type']=="padding":
                self._processed_data = self._padding(self._processed_data,preprocess['value'])
            else:
                raise Exception(preprocess['type']+" has not been implemented yet")
    def _compile_model(self):
        if 'annotation_types' in self._model_setting.keys():
            ann_types = self._model_setting['annotation_types']
        else:
            ann_types = None
        metric_types = self._model_setting['compile']['metric_types']
        loss = self._model_setting['compile']['loss']
        learning_rate=self._model_setting['compile']['learning_rate']
        values_to_ignore=self._work_setting['values_to_ignore']
        self._model_handler.compile_model(self._model,
                                          learning_rate=learning_rate,
                                          ann_types=ann_types,
                                          values_to_ignore=values_to_ignore,
                                          weights=self._weighted,
                                          metric_types=metric_types,
                                          loss_type=loss)
    def _init_worker(self):
        mode_id=self._work_setting['mode_id']
        path_root=self._work_setting['path_root']+"/"+str(self._id)+"/"+mode_id
        batch_size=self._work_setting['batch_size']
        initial_epoch=self._work_setting['initial_epoch']
        validation_split=self._work_setting['validation_split']
        period=self._work_setting['period']
        self._worker.init_worker(path_root,self._work_setting['epoch'],batch_size,
                                 initial_epoch=initial_epoch,period=period,
                                 validation_split=validation_split,
                                 is_verbose_visible=self._is_prompt_visible,
                                 is_prompt_visible=self._is_prompt_visible)
        self._worker.data = self._processed_data
        self._worker.model = self._model
    def _before_execute(self):
        self._worker.before_work()
        self._save_setting()
    def _execute(self):
        if self._is_prompt_visible:
            print('Start working('+strftime("%Y-%m-%d %H:%M:%S",gmtime())+")")
        start_time = time()
        self._worker.work()
        end_time = time()
        time_spend = end_time - start_time
        if self._is_prompt_visible:
            print('End working(' + strftime("%Y-%m-%d %H:%M:%S",gmtime()) + ")")
            print("Spend time: " + strftime("%H:%M:%S", gmtime(time_spend)))
    def _after_execute(self):
        self._worker.after_work()
    def _create_folder(self,path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError as erro:
            if erro.errno != errno.EEXIST:
                raise
    def _setting_to_saved(self):
        saved = {}
        saved['work_setting'] = self._work_setting
        saved['model_setting'] = self._model_setting
        saved['weighted'] = self._weighted
        saved['id'] = self._id
        saved['is_prompt_visible'] = self._is_prompt_visible
        saved['work_setting_path'] = self._work_setting_path
        saved['model_setting_path'] = self._model_setting_path
        return saved
    def _save_setting(self):
        data =  self._setting_to_saved()
        mode_id=self._work_setting['mode_id']
        data['save_time'] = strftime("%Y_%b_%d", gmtime())
        path_root=self._work_setting['path_root'] + "/" + str(self._id) + "/" + mode_id
        copyfile(self._work_setting_path, path_root + "/work_setting.json")
        copyfile(self._model_setting_path, path_root + "/model_setting.json")
        with open(path_root + '/setting.json', 'w') as outfile:  
            json.dump(data, outfile)