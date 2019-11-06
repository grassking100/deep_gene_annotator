import torch.nn as nn
import torch
from abc import abstractmethod,ABCMeta,abstractproperty
from .loss import CCELoss,bce_loss,mean_by_mask
from .utils import get_seq_mask
from .inference import basic_inference

def _evaluate(loss,model,inputs, labels,lengths,inference):
    model.train(False)
    with torch.no_grad():
        outputs,lengths = model(inputs,lengths=lengths)
        masks = get_seq_mask(lengths)
        predict_result = inference(outputs, masks)
        outputs = outputs.float()
        loss_ = loss(outputs, labels, masks,predict_result=predict_result).item()
    return loss_,predict_result,lengths,masks

def _predict(model,inputs,lengths,inference):
    model.train(False)
    with torch.no_grad():
        outputs,lengths = model(inputs,lengths=lengths)
        masks = get_seq_mask(lengths)
        outputs = outputs.float()
        outputs = inference(outputs,masks)
    return outputs,lengths,masks

class IExecutor(metaclass=ABCMeta):
    @abstractmethod
    def fit(self,**kwargs):
        pass
    @abstractmethod
    def evaluate(self,**kwargs):
        pass
    @abstractmethod
    def predict(self,**kwargs):
        pass
    @abstractmethod
    def process(self,**kwargs):
        pass
    @abstractmethod
    def get_config(self,**kwargs):
        pass
    @abstractproperty
    def optimizer(self):
        pass
    @abstractmethod
    def optimizer(self,optimizer):
        pass
    @abstractproperty
    def state_dict(self):
        pass
    @abstractmethod        
    def load_state_dict(self,state_dicts):
        pass
    @abstractmethod        
    def on_epoch_end(self,**kwargs):
        pass

class _Executor(IExecutor):
    def __init__(self):
        self.loss = CCELoss()
        self.inference = basic_inference(3)

    def get_config(self,**kwargs):
        config = {}
        config['loss_config'] = self.loss.get_config()
        config['inference'] = self.inference.__name__ 
        return config

    def predict(self,model,inputs,lengths,**kwargs):
        outputs,lengths,masks = _predict(model,inputs,lengths,
                                         inference=self.inference,**kwargs)
        return outputs,lengths,masks
    
    def evaluate(self,model,inputs,labels,lengths,**kwargs):
        if self.loss is not None:
            loss, outputs,lengths,masks = _evaluate(self.loss,model,inputs,labels,lengths,
                                                    inference=self.inference,**kwargs)
            return {'loss':loss},outputs,lengths,masks
        else:
            outputs,lengths,masks = _predict(model,inputs,lengths,inference=self.inference,**kwargs)
            return {},outputs,lengths,masks

class BasicExecutor(_Executor):
    def __init__(self):
        super().__init__()
        self.grad_clip = None
        self.grad_norm = None
        self._optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_target = 'loss'
        self._lr_history = {}

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,optimizer):
        self._optimizer = optimizer
        
    def state_dict(self):
        state_dict = {'optimizer':self.optimizer.state_dict()}
        state_dict['lr_history'] = self._lr_history
        if self.lr_scheduler is not None:
            state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        return state_dict
        
    def load_state_dict(self,state_dicts):
        self.optimizer.load_state_dict(state_dicts['optimizer'])
        self._lr_history = state_dicts['lr_history']
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dicts['lr_scheduler'])

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['grad_clip'] = self.grad_clip
        config['grad_norm'] = self.grad_norm
        config['optimizer_name'] = self.optimizer.__class__.__name__
        config['optimizer'] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            config['lr_scheduler_name'] = self.lr_scheduler.__class__.__name__
            config['lr_scheduler'] = self.lr_scheduler.state_dict()
            config['lr_scheduler_target'] = self.lr_scheduler_target
        return config

    def fit(self,model,inputs, labels, lengths, **kwargs):
        if self.optimizer is None:
            raise Exception("Exectutor must set optimizer for fitting")

        model.train()
        self.optimizer.zero_grad()
        outputs,lengths = model(inputs,lengths=lengths)
        outputs = outputs.float()
        masks = get_seq_mask(lengths)
        predict_result = self.inference(outputs, masks)
        loss_ = self.loss(outputs, labels, masks,predict_result=predict_result, **kwargs)
        loss_.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_value_(model.parameters(),self.grad_clip)
        if self.grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(),self.grad_norm)
        self.optimizer.step()
        return {'loss':loss_.item()},predict_result,lengths,masks

    def process(self,model):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters())

    def on_epoch_end(self,epoch,metric,**kwargs):
        for index,group in enumerate(self.optimizer.param_groups):
            if index not in self._lr_history:
                self._lr_history[index] = []
            self._lr_history[index].append(group['lr'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(metric[self.lr_scheduler_target],epoch=epoch)
