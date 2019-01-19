from abc import ABCMeta, abstractmethod, abstractproperty
from os.path import expanduser
from .model_builder import ModelBuilder
from ...utils.utils import create_folder
from ...function.model_processor import IModelProcessor

class ModelCreator(IModelProcessor):
    def __init__(self,setting,weights_path=None):
        self._model = None
        self._record = {}
        self._record['setting'] = setting
        self._record['weights_path'] = weights_path
        self._setting = setting
        self._weights_path = None
        if weights_path is not None:
            self._weights_path = expanduser(weights_path)
    def process(self):
        self._model = ModelBuilder(self._setting).build()
        if self._weights_path is not None:
            self._model.load_weights(self._weights_path, by_name=True)
    @property
    def model(self):
        return self._model
    @property
    def record(self):
        return self._record
    def before_process(self,path=None):
        if path is not None:
            json_path = create_folder(path) + "/pipeline/setting/model.json"
            with open(json_path,'w') as fp:
                json.dump(self._record,fp)