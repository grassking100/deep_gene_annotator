import os
import abc
from ..utils.utils import create_folder, write_json
from ..file_process.utils import BASIC_GENE_ANN_TYPES
from .utils import MessageRecorder
from .worker import TrainWorker, BasicWorker
from .callback import CategoricalMetric
from .callback import Callbacks, ConfusionMatrix, MeanRecorder, DataHolder
from .tensorboard import TensorboardCallback
from .lr_scheduler import LRSchedulerCallback
from .model import create_model
from .executor import create_executor_builder
from .checkpoint import build_checkpoint


def _create_default_callbacks(ann_types,prefix=None):
    callbacks = Callbacks()
    metric = CategoricalMetric(len(ann_types),label_names=ann_types,prefix=prefix)
    matrix = ConfusionMatrix(len(ann_types),prefix=prefix)
    callbacks.add([metric, matrix])
    return callbacks


class Director(metaclass=abc.ABCMeta):
    def __init__(self,executor,ann_types=None,root=None):
        self._root = root
        self._ann_types = ann_types or BASIC_GENE_ANN_TYPES
        self._executor = executor
        
    def get_config(self):
        config = {}
        config['root'] = self._root
        config['ann_types'] = self._ann_types
        config['executor'] = self._executor.get_config()
        return config
    
    @abc.abstractmethod
    def execute(self):
        pass
    
    def _save_setting(self,name):
        if self._root is not None:
            settings = self.get_config()
            setting_root = os.path.join(self._root, 'settings')
            create_folder(setting_root)
            path = os.path.join(setting_root, name)
            write_json(settings, path)

            
class Trainer(Director):
    def __init__(self,train_executor,val_executor,other_executor,
                 create_tensorboard=True,epoch=None,root=None,ann_types=None):
        super().__init__(other_executor,ann_types,root)
        self._train_executor = train_executor
        self._val_executor = val_executor
        self._epoch = epoch or 1
        self._create_tensorboard = create_tensorboard
        self._train_executor.callbacks = self._create_default_train_callbacks().add(self._train_executor.callbacks)
        self._val_executor.callbacks = self._create_default_val_callbacks().add(self._train_executor.callbacks)
        self._executor.callbacks = self._create_default_other_callbacks().add(self._executor.callbacks)
        self._save_setting('trainer_config.json')
        
    def _create_default_callbacks(self,prefix=None):
        callbacks = _create_default_callbacks(self._ann_types,prefix)
        return callbacks
        
    def get_config(self):
        config = super().get_config()
        config['epoch'] = self._epoch
        config['create_tensorboard'] = self._create_tensorboard
        config['train_executor'] = self._train_executor.get_config()
        config['val_executor'] = self._val_executor.get_config()
        return config
        
    def _create_default_train_callbacks(self):
        callbacks = self._create_default_callbacks()
        callbacks.add(MeanRecorder())
        if self._create_tensorboard and self._root is not None:
            callbacks.add(TensorboardCallback(os.path.join(self._root,'train'),'train'))
        return callbacks
    
    def _create_default_val_callbacks(self):
        callbacks = self._create_default_callbacks('val')
        callbacks.add(DataHolder(prefix='val'))
        if self._create_tensorboard and self._root is not None:
            callbacks.add(TensorboardCallback(os.path.join(self._root,'val'),'val'))
        return callbacks
    
    def _create_default_other_callbacks(self):
        callbacks = Callbacks()
        if self._train_executor.lr_scheduler is not None:
            lr_scheduler_callbacks = LRSchedulerCallback(self._train_executor.lr_scheduler)
            callbacks.add(lr_scheduler_callbacks)
        return callbacks
        
    def execute(self):
        message_recorder = None
        if self._root is not None:
            message_recorder = MessageRecorder(path=os.path.join(self._root, "message.txt"))
        worker = TrainWorker(self._train_executor,self._val_executor,self._executor,
                             epoch=self._epoch,message_recorder=message_recorder)
        worker.work()
        return worker

    
class Predictor(Director):
    def __init__(self,predict_executor,root=None,ann_types=None):
        super().__init__(predict_executor,ann_types,prefix='predict',root=root)
        self._save_setting('predictor_config.json')

    def execute(self):
        worker = BasicWorker(self._executor)
        worker.work()
        return worker

    
class Tester(Director):
    def __init__(self,test_executor,root=None,prefix=None,ann_types=None):
        super().__init__(test_executor,ann_types,root)
        self._prefix = prefix or 'test'
        self._executor.callbacks = self._create_default_callbacks().add(self._executor.callbacks)
        self._save_setting('tester_config.json')
        
    def _create_default_callbacks(self,prefix=None):
        prefix = prefix or self._prefix
        callbacks =  _create_default_callbacks(self._ann_types,prefix)
        callbacks.add(DataHolder(prefix=prefix))
        if self._root is not None:
            checkpoint = build_checkpoint(self._root,self._prefix,force_reset=True)
            callbacks.add(checkpoint)
        return callbacks
            
    def execute(self):
        worker = BasicWorker(self._executor)
        worker.work()
        return worker


def create_model_exe_builder(model_settings_path,excutor_settings_path,
                             model_weights_path=None,executor_weights_path=None,
                             save_distribution=False):
    # Create model
    model = create_model(model_settings_path,
                         weights_path=model_weights_path,
                         save_distribution=save_distribution)
    # Create exe_builder
    exe_builder = create_executor_builder(excutor_settings_path,executor_weights_path)
    return model, exe_builder


def create_best_model_exe_builder(saved_root,latest=False):
    setting = read_json(os.path.join(saved_root, 'train_main_setting.json'))
    resource_root = os.path.join(saved_root,'resource')
    checkpoint_root = os.path.join(saved_root,'checkpoint')
    exe_file_name = get_file_name(setting['executor_settings_path'], True)
    model_file_name = get_file_name(setting['model_settings_path'], True)
    executor_settings_path = os.path.join(resource_root,exe_file_name)
    model_settings_path = os.path.join(resource_root,model_file_name)
    if latest:
        model_weights_path = os.path.join(checkpoint_root,'latest_model.pth')
    else:
        model_weights_path = os.path.join(checkpoint_root,'best_model.pth')
    model, executor = create_model_exe_builder(model_settings_path,executor_settings_path,
                                            model_weights_path=model_weights_path)
    return model, executor

