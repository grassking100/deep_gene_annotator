from .loss import CCELoss
import torch.nn as nn
import torch
from abc import abstractmethod

bce_loss = nn.BCELoss(reduction='mean')

def _evaluate(loss,model,inputs, labels,lengths,inference=None,**kwargs):
    model.train(False)
    with torch.no_grad():
        outputs = model(inputs,lengths=lengths)
        loss_ = loss(outputs, labels,**kwargs)
        if inference is not None:
            outputs = inference(outputs)
    return loss_.item(),outputs

def _predict(model,inputs,lengths,inference=None,**kwargs):
    model.train(False)
    with torch.no_grad():
        outputs = model(inputs,lengths=lengths)
        if inference is not None:
            outputs = inference(outputs)
    return outputs

class IExecutor:
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

class _Executor(IExecutor):
    def __init__(self):
        self.loss = CCELoss()
        self.inference = None

    def get_config(self,**kwargs):
        config = {}
        config['loss'] = self.loss
        config['inference'] = self.inference
        return config

    def evaluate(self,model,inputs,labels,lengths,**kwargs):
        loss, outputs = _evaluate(self.loss,model,inputs,labels,lengths,self.inference,**kwargs)
        return {'loss':loss},outputs
    
    def predict(self,model,inputs,lengths,**kwargs):
        return _predict(model,inputs,lengths,self.inference,**kwargs)
    
class BasicExecutor(_Executor):
    def __init__(self):
        self.grad_clip = None
        self.grad_norm = None
        self.optimizer = None

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['grad_clip'] = self.grad_clip
        config['grad_norm'] = self.grad_norm
        config['optimizer'] = self.optimizer.state_dict()
        
    def fit(self,model,inputs, labels, lengths,**kwargs):
        if self.optimizer is None:
            raise Exception("Exectutor must set optimizer for fitting")
        model.train(True)
        self.optimizer.zero_grad()
        outputs = model(inputs,lengths=lengths)
        loss_ = self.loss(outputs, labels, **kwargs)
        loss_.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(),self.grad_clip)
        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),self.grad_norm)
        self.optimizer.step()
        if self.inference is not None:
            outputs = self.inference(outputs)
        return {'loss':loss_.item()},outputs

    def process(self,model):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters())        
            
class GANExecutor(_Executor):
    def __init__(self):
        self.reverse_inference = None
        self.optimizer = None
        self._label_optimizer = None
        self._discrim_optimizer = None

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['reverse_inference'] = self.reverse_inference
        config['label_optimizer'] = self._label_optimizer.state_dict()
        config['discrim_optimizer'] = self._discrim_optimizer.state_dict()
        
    def fit(self,model,inputs, labels, lengths,**kwargs):
        if self._label_optimizer is None or self._discrim_optimizer is None:
            raise Exception("Exectutor must set optimizer for fitting")
        model.train(True)    
        label_model, discrim_model = model.gan,model.discrim
        
        self._discrim_optimizer.zero_grad()
        predict_labels = label_model(inputs,lengths=lengths)
        if self.reverse_inference is not None:
            labels_ = self.reverse_inference(labels)
        else:
            labels_ = labels
        real_status = discrim_model(inputs,labels_,lengths=lengths)
        labels_status = discrim_model(inputs,predict_labels,lengths=lengths)
        ls_loss = bce_loss(labels_status,torch.zeros(len(labels_status)).cuda())
        rs_loss = bce_loss(real_status,torch.ones(len(real_status)).cuda())
        loss_ = ls_loss + rs_loss
        loss_.backward()
        self._label_optimizer.step()
        self._discrim_optimizer.zero_grad()
        outputs = label_model(inputs,lengths=lengths)
        outputs_status = discrim_model(inputs,outputs,lengths=lengths)
        disrim_loss = self.loss(outputs, labels, **kwargs) + bce_loss(outputs_status,torch.ones(len(outputs_status)).cuda())
        disrim_loss.backward()
        if self.inference is not None:
            outputs = self.inference(outputs,labels)
        return {'loss':loss_.item(),'disrim_loss':disrim_loss.item()},outputs

    def evaluate(self,model,inputs,labels,lengths,**kwargs):
        return super().evaluate(model.gan,inputs,lengths,kwargs)
    
    def predict(self,model,inputs,lengths,**kwargs):
        return super().predict(model.gan,inputs,lengths,kwargs)
        
    def process(self,model):
        if self.optimizer is None:
            self._label_optimizer = torch.optim.Adam(model.gan.parameters())
            self._discrim_optimizer = torch.optim.Adam(model.discrim.parameters())
        else:
            self._label_optimizer, self._discrim_optimizer = self.optimizer
