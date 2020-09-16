import os
import abc
import torch
import torch.optim
from ..utils.utils import read_json
from ..utils.utils import print_progress
from .utils import get_seq_mask, get_name_parameter
from .loss import CCELoss
from .inference import BasicInference
from .grad_clipper import GradClipper
from .data_generator import SeqGenerator, SeqCollateWrapper, SeqDataset
from .callback import Callbacks
from .lr_scheduler import LRScheduler


def _get_params(model, target_weight_decay=None, weight_decay_name=None):
    target_weight_decay = target_weight_decay or []
    weight_decay_name = weight_decay_name or []
    if len(target_weight_decay) == len(weight_decay_name):
        params = []
        special_names = []
        for name, weight_decay_ in zip(weight_decay_name, target_weight_decay):
            returned_names, parameters = get_name_parameter(model, [name])
            special_names += returned_names
            params += [{'params': parameters, 'weight_decay': weight_decay_}]
        default_parameters = []
        for name_, parameter in model.named_parameters():
            if name_ not in special_names:
                default_parameters.append(parameter)
        params += [{'params': default_parameters}]
        return params
    else:
        raise Exception("Different number between target_weight_decay and weight_decay_name")

def create_optimizer(type_,parameters,**kwargs):
    optimizer = getattr(torch.optim,type_)
    return optimizer(parameters,**kwargs)

def _evaluate(loss, model,seq_data, inference, **kwargs):
    inputs = seq_data.get('inputs').cuda()
    labels = seq_data.get('answers').cuda()
    lengths = seq_data.get('lengths')
    model.train(False)
    with torch.no_grad():
        outputs, lengths = model(inputs, lengths=lengths)
        outputs = outputs.float()
        masks = get_seq_mask(lengths).cuda()
        loss_ = loss(outputs, labels, masks,
                     seq_data=seq_data,**kwargs).item()
        predict_result = inference(outputs)
    return {
        'loss': loss_,
        'predicts': SeqDataset({'annotation':predict_result}),
        'lengths': lengths,
        'masks': masks,
        'outputs': outputs.cpu().numpy()
    }


def _predict(model, seq_data, inference):
    inputs = seq_data.get('inputs').cuda()
    lengths=seq_data.get('lengths')
    model.train(False)
    with torch.no_grad():
        outputs, lengths = model(inputs, lengths=lengths)
        outputs = outputs.float()
        masks = get_seq_mask(lengths).cuda()
        predict_result = inference(outputs)
    return {
        'predicts': SeqDataset({'annotation':predict_result}),
        'lengths': lengths,
        'masks': masks,
        'outputs': outputs.cpu().numpy()
    }

class AbsractExecutor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def execute(self,**kwargs):
        pass

    @abc.abstractmethod
    def get_config(self):
        pass

    def on_epoch_end(self,epoch, metric=None):
        pass
    
    def get_state_dict(self):
        state_dict = {}
        return state_dict
      
    def load_state_dict(self, state_dicts):
        pass
    
    def load(self,path):
        state_dict = torch.load(path,map_location='cpu')
        self.load_state_dict(state_dict)
        
    def save(self,path,overwrite=False):
        if os.path.exists(path) and not overwrite:
            raise Exception("Try to overwrite the existed weights to {}".format(path))
        torch.save(self.get_state_dict(), path)
    
class BasicExecutor(AbsractExecutor):
    def __init__(self,model,data_generator, inference=None, callbacks=None):
        self._inference = inference or BasicInference(3)
        self._data_generator = data_generator
        self._model = model
        self._callbacks = callbacks or Callbacks()

    @property
    def model(self):
        return self._model
    
    @property
    def callbacks(self):
        return self._callbacks
    
        
    @abc.abstractmethod
    def _process(self,data):
        pass
        
    def _batch_process(self,data):
        self.callbacks.on_batch_begin()
        returned = self._process(data)
        with torch.no_grad():
            self.callbacks.on_batch_end(predicts=returned['predicts'],
                                         outputs=returned['outputs'],
                                         seq_data=data,
                                         metric=returned['metric'],
                                         masks=returned['masks'])
        torch.cuda.empty_cache()
        
    def execute(self):
        batch_info = "Processing {:.1f}% of data\n"
        for index, data in enumerate(self._data_generator):
            info = batch_info.format(100*(index+1)/len(self._data_generator))
            print_progress(info)
            self._batch_process(data)

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['inference'] = self._inference.get_config()
        config['callbacks'] = self._callbacks.get_config()
        return config
    
class TestExecutor(BasicExecutor):
    def __init__(self,model,data_generator,loss=None,inference=None, callbacks=None):
        super().__init__(model,data_generator,inference,callbacks)
        self._loss = loss or CCELoss()

    def _process(self,data):
        torch.cuda.empty_cache()
        self._model.reset()
        returned = _evaluate(self._loss,self._model,data,
                             inference=self._inference,accumulate=True)
        returned['metric'] = {'loss': returned['loss']}
        self._model.reset()
        torch.cuda.empty_cache()
        return returned
    
    def execute(self):
        super().execute()
        self._loss.reset_accumulated_data()
    
    def get_config(self):
        config = super().get_config()
        config['loss'] = self._loss.get_config()
        return config
    
class PredictExecutor(BasicExecutor):
    def _process(self,data):
        torch.cuda.empty_cache()
        self._model.reset()
        returned = _predict(self._model,data,inference=self._inference)
        returned['metric'] = {}
        self._model.reset()
        torch.cuda.empty_cache()
        return returned

class TrainExecutor(BasicExecutor):
    def __init__(self,model,data_generator,optimizer,
                 loss=None,inference=None, callbacks=None,
                 grad_clipper=None,lr_scheduler=None):
        super().__init__(model,data_generator,inference,callbacks)
        self._loss = loss or CCELoss()
        self._optimizer = optimizer
        self._has_fit = False
        self._grad_clipper = grad_clipper or GradClipper()
        self._lr_scheduler = lr_scheduler

    @property
    def optimizer(self):
        return self._optimizer
        
    @property
    def lr_scheduler(self):
        return self._lr_scheduler
    
    def _process(self,data):
        torch.cuda.empty_cache()
        inputs = data.get('inputs').cuda()
        labels = data.get('answers').cuda()
        lengths = data.get('lengths')
        self._has_fit = True
        self._model.train()
        self._optimizer.zero_grad()
        outputs, lengths = self._model(inputs, lengths=lengths, answers=labels)
        masks = get_seq_mask(lengths).cuda()
        with torch.no_grad():
            predict_result = self._inference(outputs)
        loss = self._loss(outputs,labels,masks)
        loss.backward()
        self._grad_clipper.clip(self._model.parameters())
        self._optimizer.step()
        #Caution: the training result is result of model before updated
        returned = {
            'metric': {'loss': loss.item()},
            'predicts': SeqDataset({'annotation':predict_result}),
            'lengths': lengths,
            'masks': masks,
            'outputs': outputs
        }
        torch.cuda.empty_cache()
        return returned

    def execute(self):
        super().execute()
        self._loss.reset_accumulated_data()
    
    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict['optimizer'] = self._optimizer.state_dict()
        if self._lr_scheduler is not None:
            state_dict['lr_scheduler'] = self._lr_scheduler.state_dict()
        return state_dict
      
    def load_state_dict(self, state_dicts):
        super().load_state_dict(state_dicts)
        self._optimizer.load_state_dict(state_dicts['optimizer'])
        if self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(state_dicts['lr_scheduler'])
      
    def get_config(self):
        config = super().get_config()
        config['loss'] = self._loss.get_config()
        config['grad_clipper'] = self._grad_clipper.get_config()
        config['optimizer_name'] = self._optimizer.__class__.__name__
        param_groups = []
        for group in self._optimizer.state_dict()['param_groups']:
            group = dict(group)
            del group['params']
            param_groups.append(group)
        config['optimizer'] = param_groups
        if self._lr_scheduler is not None:
            config['lr_scheduler'] = self._lr_scheduler.get_config()
        return config

class BasicExecutorBuilder:
    def __init__(self,inference=None,loss=None):
        self._inference = inference or BasicInference(3)
        self._loss = loss or CCELoss()
        self._optimizer_kwargs = {}
        self._optim_type = 'Adam'
        self._grad_clipper = None
        self._parameters = None
        self._set_lr_scheduler = False
        self._lr_scheduler_kwargs = {}
        self._weights_paths = {'train':None,'test':None,'predict':None}
        self._data_generators = {
            'train':SeqGenerator(),
            'test':SeqGenerator(),
            'predict':SeqGenerator()
        }
        
    def get_config(self):
        config = {}
        config['weights_paths'] = self._weights_paths
        config['loss'] = self._loss.get_config()
        config['inference'] = self._inference.get_config()
        config['optimizer_kwargs'] = self._optimizer_kwargs
        config['set_lr_scheduler'] = self._set_lr_scheduler
        config['lr_scheduler_kwargs'] = self._lr_scheduler_kwargs
        config['optim_type'] = self._optim_type
        config['train_data_generators'] = self._data_generators['train'].get_config()
        config['test_data_generators'] = self._data_generators['test'].get_config()
        config['predict_data_generators'] = self._data_generators['predict'].get_config()
        if self._grad_clipper is not None:
            config['grad_clipper'] = self._grad_clipper.get_config()
        return config
    
    def get_data_generator(self,type_):
        return self._data_generators[type_]
        
    def set_grad_clipper(self,**kwargs):
        self._grad_clipper = GradClipper(**kwargs)
        return self

    def set_optimizer(self, optim_type=None, parameters=None, **kwargs):
        kwargs_ = dict(kwargs)
        for key,value in kwargs.items():
            if value is None:
                del kwargs_[key]
        self._optimizer_kwargs = kwargs_
        self._optim_type = optim_type or 'Adam'
        self._parameters = parameters
        return self

    def set_lr_scheduler(self,**kwargs):
        self._set_lr_scheduler = True
        self._lr_scheduler_kwargs = kwargs
        return self
        
    def set_data_generator(self,type_,batch_size=None,drop_last=False,**kwargs):
        if type_ not in self._data_generators:
            raise
        batch_size = batch_size or 1
        self._data_generators[type_] = SeqGenerator(batch_size=batch_size,drop_last=drop_last,
                                                    seq_collate_fn=SeqCollateWrapper(**kwargs))
        return self
    
    def set_weights_path(self,type_,path):
        if type_ not in self._weights_paths:
            raise
        self._weights_paths[type_] = path
        
    def build(self, type_, model, data, callbacks=None):
        data_generator = self._data_generators[type_](data)
        if type_ == 'train':
            lr_scheduler = None
            optimizer = create_optimizer(self._optim_type,self._parameters or model.parameters(),
                                         **self._optimizer_kwargs)
            if self._set_lr_scheduler:
                lr_scheduler = LRScheduler(optimizer,**self._lr_scheduler_kwargs)
            exe = TrainExecutor(model,data_generator,optimizer,loss=self._loss,
                                inference=self._inference, callbacks=callbacks,
                                grad_clipper=self._grad_clipper,lr_scheduler=lr_scheduler)
        elif type_ == 'test':
            exe = TestExecutor(model,data_generator,loss=self._loss,
                                inference=self._inference, callbacks=callbacks)
        elif type_ == 'predict':
            exe = PredictExecutor(model,data_generator,inference=self._inference, callbacks=callbacks)
        else:
            raise
        if self._weights_paths[type_] is not None:
            exe.load(self._weights_paths[type_])
        return exe

def create_executor_builder(settings,weights_path=None):    
    if isinstance(settings, str):
        settings = read_json(settings)

    inference = loss = None
    batch_size = settings['batch_size']
    train_data_generator_kwargs = settings['train_data_generator_kwargs']
    if settings['loss_type'] is not None:
        loss = LOSS_TYPES[settings['loss_type']]
    if settings['inference_type'] is not None:
        inference = INFERENCES_TYPES[settings['inference_type']]
        
    exe_builder = BasicExecutorBuilder(loss=loss,inference=inference)
    if settings['set_lr_scheduler']:
        exe_builder.set_lr_scheduler(**settings['lr_scheduler_kwargs'])
    if settings['set_grad_clipper']:
        exe_builder.set_grad_clipper(**settings['grad_clipper_kwargs'])
    exe_builder.set_optimizer(**settings['optimizer_kwargs'])        
    exe_builder.set_data_generator('train',batch_size=batch_size,**train_data_generator_kwargs)
    exe_builder.set_data_generator('test',batch_size=batch_size)
    exe_builder.set_data_generator('predict',batch_size=batch_size)
    exe_builder.set_weights_path('train',weights_path)
    return exe_builder
