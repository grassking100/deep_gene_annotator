import os
import abc
import torch
import torch.optim
from ..utils.utils import read_json,write_json,print_progress
from .utils import get_seq_mask, get_name_parameter,param_num
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
    inputs = seq_data.get('input').cuda()
    labels = seq_data.get('answer').cuda()
    lengths = seq_data.get('length')
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
        'predict': SeqDataset({'annotation':predict_result}),
        'length': lengths,
        'mask': masks,
        'output': outputs.cpu().numpy()
    }


def predict(model, seq_data, inference):
    inputs = seq_data.get('input').cuda()
    lengths=seq_data.get('length')
    model.train(False)
    with torch.no_grad():
        outputs, lengths = model(inputs, lengths=lengths)
        outputs = outputs.float()
        masks = get_seq_mask(lengths).cuda()
        predict_result = inference(outputs)
    return {
        'predict': SeqDataset({'annotation':predict_result}),
        'length': lengths,
        'mask': masks,
        'output': outputs.cpu().numpy()
    }


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


class BasicExecutor(metaclass=abc.ABCMeta):
    def __init__(self,root=None):
        self.callbacks = Callbacks()
        self._root = root
        
    def on_work_begin(self,worker):
        self.callbacks.on_work_begin(worker=worker)
    
    def on_work_end(self):
        self.callbacks.on_work_end()
    
    def on_epoch_begin(self, counter):
        self.callbacks.on_epoch_begin(counter=counter)
    
    def on_epoch_end(self, metric):
        self.callbacks.on_epoch_end(metric=metric)
                
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['root'] = self._root
        return config

    def execute(self,**kwargs):
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
    

class AdvancedExecutor(BasicExecutor):
    def __init__(self,model,data_loader, inference=None,root=None):
        super().__init__(root)
        self._inference = inference or BasicInference(3)
        self._data_loader = data_loader
        self._model = model
        
    @abc.abstractmethod
    def _process(self,data):
        pass
        
    def _batch_process(self,data):
        self.callbacks.on_batch_begin()
        returned = self._process(data)
        with torch.no_grad():
            self.callbacks.on_batch_end(predicts=returned['predict'],
                                        outputs=returned['output'],
                                        seq_data=data,
                                        metric=returned['metric'],
                                        masks=returned['mask'])
        torch.cuda.empty_cache()
        
    def execute(self):
        batch_info = "Processing {:.1f}% of data\n"
        for index, data in enumerate(self._data_loader):
            info = batch_info.format(100*(index+1)/len(self._data_loader))
            print_progress(info)
            self._batch_process(data)

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['inference'] = self._inference.get_config()
        config['callbacks'] = self.callbacks.get_config()
        return config
    
    
class TestExecutor(AdvancedExecutor):
    def __init__(self,model,data_loader,loss=None,inference=None,root=None):
        super().__init__(model,data_loader,inference,root)
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
        self._loss.reset_accumulated_data()
        super().execute()
        self._loss.reset_accumulated_data()
    
    def get_config(self):
        config = super().get_config()
        config['loss'] = self._loss.get_config()
        return config
    

class PredictExecutor(AdvancedExecutor):
    def _process(self,data):
        torch.cuda.empty_cache()
        self._model.reset()
        returned = predict(self._model,data,inference=self._inference)
        returned['metric'] = {}
        self._model.reset()
        torch.cuda.empty_cache()
        return returned

    
class TrainExecutor(AdvancedExecutor):
    def __init__(self,model,optimizer,data_loader,loss=None,inference=None,
                 grad_clipper=None,lr_scheduler=None,root=None):
        super().__init__(model,data_loader,inference,root)
        self._loss = loss or CCELoss()
        self._optimizer = optimizer
        self._has_fit = False
        self._grad_clipper = grad_clipper or GradClipper()
        self._lr_scheduler = lr_scheduler

    def on_work_begin(self,worker):
        if self._root is not None:
            save_model_settings(os.path.join(self._root, 'settings'),self._model)
        super().on_work_begin(worker)
        
    @property
    def optimizer(self):
        return self._optimizer
        
    @property
    def lr_scheduler(self):
        return self._lr_scheduler
    
    def _process(self,data):
        torch.cuda.empty_cache()
        inputs = data.get('input').cuda()
        labels = data.get('answer').cuda()
        lengths = data.get('length')
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
        returned = {'metric': {'loss': loss.item()},
                    'predict': SeqDataset({'annotation':predict_result}),
                    'length': lengths,'mask': masks,'output': outputs}
        torch.cuda.empty_cache()
        return returned

    def execute(self):
        self._loss.reset_accumulated_data()
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

    
class AdvExeBuilder:
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
            'train':SeqGenerator(),'test':SeqGenerator(),'predict':SeqGenerator()
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
        
    def set_data_generator(self,type_,batch_size=None,drop_last=False,shuffle=False,**seq_collate_kwargs):
        if type_ not in self._data_generators:
            raise
        batch_size = batch_size or 1
        self._data_generators[type_] = SeqGenerator(batch_size=batch_size,drop_last=drop_last,shuffle=shuffle,
                                                    seq_collate_fn=SeqCollateWrapper(**seq_collate_kwargs))
        return self
    
    def set_weights_path(self,type_,path):
        if type_ not in self._weights_paths:
            raise
        self._weights_paths[type_] = path
        
    def build(self, type_, model, data ,root=None):
        data_loader = self._data_generators[type_](data)
        if type_ == 'train':
            lr_scheduler = None
            optimizer = create_optimizer(self._optim_type,self._parameters or model.parameters(),
                                         **self._optimizer_kwargs)
            if self._set_lr_scheduler:
                lr_scheduler = LRScheduler(optimizer,**self._lr_scheduler_kwargs)
            exe = TrainExecutor(model,optimizer,data_loader=data_loader, loss=self._loss,
                                inference=self._inference, grad_clipper=self._grad_clipper,
                                lr_scheduler=lr_scheduler,root=root)
        elif type_ == 'test':
            exe = TestExecutor(model,data_loader,loss=self._loss,
                               inference=self._inference,root=root)
        elif type_ == 'predict':
            exe = PredictExecutor(model,data_loader, inference=self._inference,root=root)
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
    seq_collate_kwargs = settings['seq_collate_kwargs']
    if settings['loss_type'] is not None:
        loss = LOSS_TYPES[settings['loss_type']]
    if settings['inference_type'] is not None:
        inference = INFERENCES_TYPES[settings['inference_type']]
        
    exe_builder = AdvExeBuilder(loss=loss,inference=inference)
    if settings['set_lr_scheduler']:
        exe_builder.set_lr_scheduler(**settings['lr_scheduler_kwargs'])
    if settings['set_grad_clipper']:
        exe_builder.set_grad_clipper(**settings['grad_clipper_kwargs'])
    exe_builder.set_optimizer(**settings['optimizer_kwargs'])        
    exe_builder.set_data_generator('train',batch_size=batch_size,
                                   shuffle=True,drop_last=True,
                                   **seq_collate_kwargs)
    exe_builder.set_data_generator('test',batch_size=batch_size)
    exe_builder.set_data_generator('predict',batch_size=batch_size)
    exe_builder.set_weights_path('train',weights_path)
    return exe_builder
