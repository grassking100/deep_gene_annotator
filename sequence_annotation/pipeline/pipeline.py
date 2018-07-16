"""This submodule provides class to defined Pipeline"""
import os
import errno
import json
from abc import ABCMeta, abstractmethod
from time import strftime, gmtime, time
from os.path import expanduser,abspath
from . import ModelHandler
from . import SimpleDataHandler
class Pipeline(metaclass=ABCMeta):
    def __init__(self, is_prompt_visible=True):
        self._id = None
        self._is_prompt_visible = is_prompt_visible
        self._worker = None
        self._work_setting = None
        self._model_setting = None
        self._preprocessed_data = None
        self._processed_data = None
        self._class_weight = None
        self._model = None
        self._dynamic_weight_method = None
        self.data_handler = SimpleDataHandler()
        self.model_handler = ModelHandler()
    @property
    def worker(self):
        return self._worker
    def print_prompt(self,value):
        if self._is_prompt_visible:
            print(value)
    def execute(self,id_, work_setting,model_setting):
        self._id = id_
        self._work_setting = work_setting
        self._model_setting = model_setting
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
        pass
    def _init_model(self):
        load_model = self._work_setting['load_model']
        if load_model:
            model_path = expanduser(self._work_setting['trained_model_path'])
            self.print_prompt("\tLoading model from "+model_path)
            self._model = self.model_handler.load_model(model_path)
        else:
            self.print_prompt("\tBuilding model...")
            self._model = self.model_handler.build_model(self._model_setting)
            load_weights = self._work_setting['load_weights']
            if load_weights:
                model_path = expanduser(self._work_setting['trained_model_path'])
                self.print_prompt("\tLoading weights from "+model_path)
                self._model.load_weights(model_path, by_name=True)
    def _padding(self, data_dict, value):
        padded = {}
        for data_kind, data in data_dict.items():
            inputs = data['inputs']
            answers = data['answers']
            inputs, answers = self.data_handler.padding(inputs,answers,value)
            padded[data_kind]= {"inputs":inputs,"answers":answers}
        return padded
    def _process_data(self):
        self._processed_data = {}
        for data_kind,data in self._preprocessed_data.items():
            inputs,answers = self.data_handler.to_vecs(data['data_pair'].values())
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
        metric_types = self._work_setting['compile']['metric_types']
        loss = self._work_setting['compile']['loss']
        learning_rate=self._work_setting['compile']['learning_rate']
        values_to_ignore=self._work_setting['values_to_ignore']
        self.model_handler.compile_model(self._model,learning_rate=learning_rate,
                                         ann_types=ann_types,
                                         values_to_ignore=values_to_ignore,
                                         class_weight=self._class_weight,
                                         metric_types=metric_types,loss_type=loss,
                                         dynamic_weight_method=self._dynamic_weight_method)
    @abstractmethod
    def _init_worker(self):
        pass
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
                print_prompt("Create folder:"+path)
                os.makedirs(path)
        except OSError as erro:
            if erro.errno != errno.EEXIST:
                raise
    def _load_data(self):
        self._preprocessed_data = {}
        data_path =  self._work_setting['data_path']
        ann_types =  self._model_setting['annotation_types']
        input_setting =  self._model_setting['input_setting']
        for name,path in data_path.items():
            temp = self.data_handler.get_data(path['inputs'],
                                              path['answers'],
                                              ann_types=ann_types,
                                              setting = input_setting)
            self._preprocessed_data[name] = temp
    def _setting_to_saved(self):
        saved = {}
        saved['model_parameter_numer'] = self._model.count_params()
        saved['work_setting'] = self._work_setting
        saved['model_setting'] = self._model_setting
        saved['class_weight'] = self._class_weight
        saved['id'] = self._id
        saved['is_prompt_visible'] = self._is_prompt_visible
        return saved
    @abstractmethod
    def _save_setting(self):
        pass
