import os
import abc
import numpy as np
import torch
from ..utils.utils import create_folder, write_json
from ..utils.seq_converter import SeqConverter
from ..genome_handler.ann_seq_processor import seq2vecs
from .utils import param_num, MessageRecorder
from .worker import TrainWorker, BasicWorker
from .callback import CategoricalMetric
from .callback import Callbacks, ConfusionMatrix, MeanRecorder, DataHolder
from .tensorboard import TensorboardCallback, SeqFigCallback
from .lr_scheduler import LearningRateHolder,LRSchedulerCallback
from .data_generator import SeqGenerator
from .model import create_model
from .executor import create_executor_builder,BasicExecutorBuilder
from .checkpoint import build_checkpoint


def save_model_settings(root,model):
    model_config_path = os.path.join(root, "model_config.json")
    model_component_path = os.path.join(root, "model_component.txt")
    param_num_path = os.path.join(root, 'model_param_num.txt')
    if not os.path.exists(model_config_path):
        write_json(model.get_config(), model_config_path)
    if not os.path.exists(model_component_path):
        with open(model_component_path, "w") as fp:
            fp.write(str(model))
    if not os.path.exists(param_num_path):
        with open(param_num_path, "w") as fp:
            fp.write("Required-gradient parameters number:{}".format(param_num(model)))

def create_categorical_metric(ann_types, prefix=None):
    metric = CategoricalMetric(len(ann_types),label_names=ann_types,prefix=prefix)
    return metric

def create_confusion_matrix(ann_types, prefix=None):
    metric = ConfusionMatrix(len(ann_types),prefix=prefix)
    return metric

def create_data_gen(batch_size,collate_fn=None,shuffle=True,*args, **kwargs):
    train_gen = SeqGenerator(batch_size=batch_size,shuffle=shuffle,
                             seq_collate_fn=collate_fn,*args, **kwargs)
    return train_gen

def create_basic_data_gen(batch_size):
    test_gen = SeqGenerator(batch_size=batch_size,shuffle=False)
    return test_gen

def get_seq_fig(writer,ann_types, seq, ann_seq, color_settings, prefix=None):
    seq = SeqConverter().seq2vecs(seq)
    seq = np.transpose(np.array([seq]), [0, 2, 1])
    seq = torch.from_numpy(seq).type('torch.FloatTensor').cuda()
    ann_seq = [seq2vecs(ann_seq, ann_types)]
    colors = [color_settings[type_] for type_ in ann_types]
    seq_fig = SeqFigCallback(writer,seq,ann_seq,prefix=prefix,
                             label_names=ann_types,colors=colors)
    return seq_fig
    
def _create_default_callbacks(ann_types,prefix=None):
    callbacks = Callbacks()
    metric = create_categorical_metric(ann_types,prefix=prefix)
    matrix = create_confusion_matrix(ann_types,prefix=prefix)
    callbacks.add([metric, matrix])
    return callbacks

class Director(metaclass=abc.ABCMeta):
    def __init__(self,ann_types,executor_builder=None,root=None):
        self._root = root
        self._ann_types = ann_types
        self._executor_builder = executor_builder or BasicExecutorBuilder()
        
    def get_config(self):
        config = {}
        config['root'] = self._root
        config['ann_types'] = self._ann_types
        config['executor_builder'] = self._executor_builder.get_config()
        return config
    
    @abc.abstractmethod
    def execute(self):
        pass
    
    def _save_setting(self,name,settings):
        if self._root is not None:
            settings['director'] = self.get_config()
            setting_root = os.path.join(self._root, 'settings')
            create_folder(setting_root)
            path = os.path.join(setting_root, name)
            write_json(settings, path)

            
class Trainer(Director):
    def __init__(self,ann_types,model,train_data,val_data,
                 executor_builder=None,create_tensorboard=True,
                 other_callbacks=None,epoch=None,root=None,
                 checkpoint_kwargs=None):
        super().__init__(ann_types,executor_builder,root)
        self._model = model
        self._train_data = train_data
        self._val_data = val_data
        self._epoch = epoch or 1
        self._create_tensorboard = create_tensorboard
        self._checkpoint_kwargs = checkpoint_kwargs or {}
        self._train_callbacks = self._create_default_train_callbacks()
        self._val_callbacks = self._create_default_val_callbacks()
        self._other_callbacks = self._create_default_other_callbacks()
        if other_callbacks is not None:
            self._other_callbacks.add(other_callbacks)
        
    def _create_default_callbacks(self,prefix=None):
        callbacks = _create_default_callbacks(self._ann_types,prefix)
        return callbacks
        
    def get_config(self):
        config = super().get_config()
        config['epoch'] = self._epoch
        config['create_tensorboard'] = self._create_tensorboard
        return config
        
    def _create_default_train_callbacks(self):
        callbacks = self._create_default_callbacks()
        callbacks.add(MeanRecorder())
        if self._create_tensorboard and self._root is not None:
            callbacks.add(TensorboardCallback(os.path.join(self._root,'train'),'train'))
        callbacks.add(LearningRateHolder())
        return callbacks
    
    def _create_default_val_callbacks(self):
        callbacks = self._create_default_callbacks('val')
        callbacks.add(DataHolder(prefix='val'))
        if self._create_tensorboard and self._root is not None:
            callbacks.add(TensorboardCallback(os.path.join(self._root,'val'),'val'))
        return callbacks
    
    def _create_default_other_callbacks(self):
        callbacks = Callbacks()
        if self._root is not None:
            checkpoint_root = os.path.join(self._root, 'checkpoint')
            checkpoint = build_checkpoint(checkpoint_root,'train',
                                          **self._checkpoint_kwargs)
            callbacks.add(checkpoint)
        return callbacks
        
    def execute(self):
        message_recorder = None
        other_callbacks = self._other_callbacks
        train_executor = self._executor_builder.build('train',self._model,self._train_data,self._train_callbacks)
        if train_executor.lr_scheduler is not None:
            lr_scheduler_callbacks = LRSchedulerCallback(train_executor.lr_scheduler)
            other_callbacks = Callbacks([lr_scheduler_callbacks]).add(other_callbacks)
        val_executor = self._executor_builder.build('test',self._model,self._val_data,self._val_callbacks)
        settings = {
            'train_executor':train_executor.get_config(),
            'val_executor':val_executor.get_config(),
            'other_callbacks':other_callbacks.get_config()
        }
        self._save_setting("train_setting.json",settings)
        
        if self._root is not None:
            save_model_settings(os.path.join(self._root, 'settings'),train_executor.model)
            message_recorder = MessageRecorder(path=os.path.join(self._root, "message.txt"))
        worker = TrainWorker(train_executor,val_executor,epoch=self._epoch,
                             other_callbacks=other_callbacks,
                             message_recorder=message_recorder)
        worker.work()
        return worker


class BasicDirector(Director):
    def __init__(self,ann_types,model,data,root=None,prefix=None,executor_builder=None,callbacks=None):
        super().__init__(ann_types,executor_builder,root)
        self._model = model
        self._data = data
        self._prefix = prefix or 'test'
        self._callbacks = self._create_default_callbacks()
        if callbacks is not None:
            self._callbacks.add(callbacks)
        
    def _create_default_callbacks(self,prefix=None):
        prefix = prefix or self._prefix
        callbacks = Callbacks()
        callbacks.add(DataHolder(prefix=prefix))
        if self._root is not None:
            checkpoint = build_checkpoint(self._root,self._prefix,only_recorder=True,force_reset=True)
            callbacks.add(checkpoint)
        return callbacks
            
    def execute(self):
        executor = self._executor_builder.build(self._prefix,self._model,self._data,self._callbacks)
        settings = {'executor':executor.get_config()}
        self._save_setting("{}_setting.json".format(self._prefix),settings)
        save_model_settings(os.path.join(self._root, 'settings'),executor.model)
        worker = BasicWorker(executor)
        worker.work()
        return worker

    
class Tester(BasicDirector):
    def __init__(self,ann_types,model,data,**kwargs):
        super().__init__(ann_types,model,data,prefix='test',**kwargs)
        
    def _create_default_callbacks(self,prefix=None):
        prefix = prefix or self._prefix
        callbacks =  _create_default_callbacks(self._ann_types,prefix)
        callbacks.add(super()._create_default_callbacks())
        callbacks.add(DataHolder(prefix=prefix))
        return callbacks
    

class Predictor(Director):
    def __init__(self,ann_types,model,data,**kwargs):
        super().__init__(ann_types,model,data,prefix='predict',**kwargs)
        

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
