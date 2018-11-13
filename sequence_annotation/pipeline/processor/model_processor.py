from abc import ABCMeta, abstractmethod, abstractproperty
from os.path import expanduser
from ...model.model_handler import ModelHandler
from ...utils.helper import create_folder

class IModelProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self):
        pass
    @abstractproperty
    def model(self):
        pass
    @abstractproperty
    def record(self):
        pass
    def before_process(self,path=None):
        pass
    def after_process(self,path=None):
        pass

class SimpleModel(IModelProcessor):
    def __init__(self,model):
        self._record = {}
        self._model = model
    def process(self):
        pass
    @property
    def model(self):
        return self._model
    @property
    def record(self):
        return self._record

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
        self._model = ModelHandler().build_model(self._setting)
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
            json_path = create_folder(path) + "/setting/model.json"
            with open(json_path,'w') as fp:
                json.dump(self._record,fp)
                
class ModelLoader(IModelProcessor):
    def __init__(self,path):
        self._model = None
        self._record = {}
        self._record['path'] = path
        self._path = expanduser(path)
    def process(self):
        self._model = self.model_handler.load_model(self._path)
    @property
    def model(self):
        return self._model
    @property
    def record(self):
        return self._record
    def before_process(self,path=None):
        if path is not None:
            json_path = create_folder(path) + "/setting/model.json"
            with open(json_path,'w') as fp:
                json.dump(self._record,fp)