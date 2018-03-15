"""This submodule provides class to defined Pipeline"""
import os
import errno
from abc import ABCMeta, abstractmethod
from keras.models import load_model
from keras.optimizers import Adam
from . import CustomObjectsFacade
from . import ModelBuildFacade
class Pipeline(metaclass=ABCMeta):
    def __init__(self, setting_file, worker):
        self._setting_file = setting_file
        self._worker = worker
        self._setting = None
        self._model = None
        self._custom_objects = None
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
        model_build_facade = ModelBuildFacade(self._setting)
        self._model = model_build_facade.model()
    def _compile_model(self):
        optimizer = Adam(lr=self._setting['learning_rate'])
        custom_metrics = []
        not_include_keys = ["loss"]
        for key,value in self._custom_objects.items():
            if key not in not_include_keys:
                custom_metrics.append(value)
        self._model.compile(optimizer=optimizer, loss=self._custom_objects['loss'],
                            metrics=custom_metrics,sample_weight_mode="temporal")
    def _init_custom_objects(self):
        facade = CustomObjectsFacade(annotation_types = self._setting['ANN_TYPES'],
                                     values_to_ignore = self._setting['terminal_signal'],
                                     weights = self._weights)
        self._custom_objects = facade.custom_objects
    def _load_model(self):
        previous_status_root = self._setting['previous_model_file']
        self._model = load_model(previous_status_root+'.h5', self._custom_objects,compile=False)
    def _create_folder(self,path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError as erro:
            if erro.errno != errno.EEXIST:
                raise
