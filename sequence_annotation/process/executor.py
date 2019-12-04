import warnings
from abc import abstractmethod,ABCMeta,abstractproperty
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import get_seq_mask
from .loss import CCELoss,bce_loss,mean_by_mask,SeqAnnLoss, FocalLoss, LabelLoss
from .inference import basic_inference,seq_ann_inference, seq_ann_reverse_inference

OPTIMIZER_CLASS = {'Adam':optim.Adam,'SGD':optim.SGD,'AdamW':optim.AdamW,'RMSprop':optim.RMSprop}

def optimizer_generator(type_,parameters,momentum=None,nesterov=False,amsgrad=False,adam_betas=None,**kwargs):
    momentum = momentum or 0
    adam_betas = adam_betas or [0.9,0.999]
    if type_ not in OPTIMIZER_CLASS:
        raise Exception("Optimizer should be {}, but got {}".format(OPTIMIZER_CLASS,type_))
        
    #filter_ = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = OPTIMIZER_CLASS[type_]
    if type_ == 'AdamW':
        warnings.warn("\n!!!\n\nAdamW's weight decay is implemented differnetly in paper and pytorch 1.3.0, please see https://github.com/pytorch/pytorch/pull/21250#issuecomment-520559993 for more information!\n\n!!!\n")
        
    if optimizer in [optim.Adam,optim.AdamW]:
        if momentum > 0 or nesterov:
            raise
        return optimizer(parameters,amsgrad=amsgrad,betas=adam_betas,**kwargs)
    elif optimizer in [optim.RMSprop]:
        if nesterov or amsgrad:
            raise
        return optimizer(parameters,momentum=momentum,**kwargs)
    else:
        if amsgrad:
            raise
        return optimizer(parameters,momentum=momentum,
                          nesterov=nesterov,**kwargs)

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
        
class ExecutorBuilder:
    def __init__(self,use_naive=True,label_num=None,grad_clip=None,grad_norm=None,
                 predict_label_num=None,answer_label_num=None,output_label_num=None):
        self.executor = BasicExecutor()
        self.executor.grad_clip = grad_clip
        self.executor.grad_norm = grad_norm
        self.executor.loss = None
        self.use_naive=use_naive
        if self.use_naive:
            self.output_label_num = self.predict_label_num = self.answer_label_num = label_num or 3
            self.executor.inference = basic_inference(self.output_label_num)
        else:
            self.predict_label_num = predict_label_num or 2
            self.answer_label_num = answer_label_num or 3
            self.output_label_num = output_label_num or 3
            self.executor.inference = seq_ann_inference
        
    def set_optimizer(self,parameters,optim_type,learning_rate=None,reduce_lr_on_plateau=False,**kwargs):
        learning_rate = learning_rate or 1e-3
        self.executor.optimizer = optimizer_generator(optim_type,parameters,lr=learning_rate,**kwargs)
        if reduce_lr_on_plateau:
            self.executor.lr_scheduler = ReduceLROnPlateau(self.executor.optimizer,
                                                           verbose=True,threshold=0.1)
   
    def set_loss(self,gamma=None,intron_coef=None,other_coef=None,nontranscript_coef=None,
                 transcript_answer_mask=True,transcript_output_mask=False,mean_by_mask=False):
        if self.use_naive:
            loss = FocalLoss(gamma)
        else:    
            loss = SeqAnnLoss(intron_coef=intron_coef,other_coef=other_coef,
                              nontranscript_coef=nontranscript_coef,
                              transcript_answer_mask=transcript_answer_mask,
                              transcript_output_mask=transcript_output_mask,
                              mean_by_mask=mean_by_mask)
        label_loss = LabelLoss(loss)
        label_loss.predict_inference = basic_inference(self.predict_label_num)
        label_loss.answer_inference = basic_inference(self.answer_label_num)
        self.executor.loss = label_loss
        
    def build(self):
        return self.executor
            