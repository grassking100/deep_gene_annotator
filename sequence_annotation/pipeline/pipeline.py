"""This submodule provides class to defined Pipeline"""
import os
import errno
from abc import ABCMeta, abstractmethod
from keras.models import load_model
from . import CustomObjectsFacade
from . import ModelBuildFacade
class Pipeline(metaclass=ABCMeta):
    def __init__(self, setting_file, worker):
        self._setting_file = setting_file
        self._worker = worker
        self._setting = None
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
        model_build_facade = ModelBuildFacade(self._setting)
        self._model = model_build_facade.model()
    def _load_model(self):
        previous_status_root = self._setting['previous_status_root']
        facade = CustomObjectsFacade(annotation_types = self._setting['ANN_TYPES'],
                                     terminal_signal = self._setting['terminal_signal'],
                                     weights = self._setting['weights'])
        self._model = load_model(previous_status_root+'.h5', facade.custom_objects)
    def _calculate_weights(self,count):
        weights = []
        total_count = 0
        for key in self._setting['ANN_TYPES']:
            weights.append(1/count[key])
            total_count += count[key]
        sum_weight = sum(weights)
        return [weight/sum_weight for weight in weights]
    def _create_folder(self,path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError as erro:
            if erro.errno != errno.EEXIST:
                raise
