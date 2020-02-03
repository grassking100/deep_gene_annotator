import warnings
from abc import abstractmethod,ABCMeta,abstractproperty
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import get_seq_mask
from .loss import CCELoss, FocalLoss, LabelLoss, SeqAnnLoss
from .inference import create_basic_inference,seq_ann_inference

OPTIMIZER_CLASS = {'Adam':optim.Adam,'SGD':optim.SGD,'AdamW':optim.AdamW,'RMSprop':optim.RMSprop}

def optimizer_generator(type_,parameters,momentum=None,nesterov=False,
                        amsgrad=False,adam_betas=None,**kwargs):
    momentum = momentum or 0
    adam_betas = adam_betas or [0.9,0.999]
    if type_ not in OPTIMIZER_CLASS:
        raise Exception("Optimizer should be {}, but got {}".format(OPTIMIZER_CLASS,type_))

    optimizer = OPTIMIZER_CLASS[type_]
    if type_ == 'AdamW':
        warnings.warn("\n!!!\n\nAdamW's weight decay is implemented differnetly"
                      "in paper and pytorch 1.3.0, please see https://github.com"
                      "/pytorch/pytorch/pull/21250#issuecomment-520559993 for more "
                      "information!\n\n!!!\n")
        
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

def _evaluate(loss,model,inputs, labels,lengths,inference,**kwargs):
    model.train(False)
    with torch.no_grad():
        outputs,lengths = model(inputs,lengths=lengths)
        outputs = outputs.float()
        masks = get_seq_mask(lengths)
        loss_ = loss(outputs, labels, masks,**kwargs).item()
        predict_result = inference(outputs,masks)
    return loss_,predict_result,lengths,masks,outputs.cpu().numpy()

def _predict(model,inputs,lengths,inference):
    model.train(False)
    with torch.no_grad():
        outputs,lengths = model(inputs,lengths=lengths)
        outputs = outputs.float()
        masks = get_seq_mask(lengths)
        predict_result = inference(outputs,masks)
    return predict_result,lengths,masks,outputs.cpu().numpy()

class IExecutor(metaclass=ABCMeta):
    @abstractmethod
    def fit(self,model,inputs, labels, lengths, **kwargs):
        pass
    @abstractmethod
    def evaluate(self,model,inputs,labels,lengths,**kwargs):
        pass
    @abstractmethod
    def predict(self,model,inputs,lengths):
        pass
    @abstractmethod
    def get_config(self):
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
    def on_epoch_end(self,epoch,metric):
        pass

class _Executor(IExecutor):
    def __init__(self):
        self.loss = CCELoss()
        self.inference = create_basic_inference(3)

    def get_config(self):
        config = {}
        if self.loss is not None:
            config['loss_config'] = self.loss.get_config()
        config['inference'] = self.inference.__name__ 
        return config

    def predict(self,model,inputs,lengths):
        returned = _predict(model,inputs,lengths,inference=self.inference)
        return returned
    
    def evaluate(self,model,inputs,labels,lengths,**kwargs):
        if self.loss is not None:
            returned = _evaluate(self.loss,model,inputs,labels,lengths,
                                 inference=self.inference,**kwargs)

            loss_,predict_result,lengths,masks,outputs = returned
            return ({'loss':loss_}),predict_result,lengths,masks,outputs
        else:
            returned = _predict(model,inputs,lengths,inference=self.inference)
            predict_result,lengths,masks,outputs = returned
            return {},predict_result,lengths,masks,outputs

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
        state_dict = {}
        if self.optimizer is not None:
            optimizer_status = dict(self.optimizer.state_dict())
            state_dict['optimizer'] = optimizer_status
        state_dict['lr_history'] = self._lr_history
        if self.lr_scheduler is not None:
            state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict
        
    def load_state_dict(self,state_dicts):
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dicts['optimizer'])
        self._lr_history = state_dicts['lr_history']
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dicts['lr_scheduler'])

    def get_config(self):
        config = super().get_config()
        config['grad_clip'] = self.grad_clip
        config['grad_norm'] = self.grad_norm
        if self.optimizer is not None:
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
        outputs,lengths = model(inputs,lengths=lengths,answers=labels)
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
        return {'loss':loss_.item()},predict_result,lengths,masks,outputs

    def on_epoch_end(self,epoch,metric):
        self.loss.reset_accumulated_data()
        for index,group in enumerate(self.optimizer.param_groups):
            if index not in self._lr_history:
                self._lr_history[index] = []
            self._lr_history[index].append(group['lr'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(metric[self.lr_scheduler_target],epoch=epoch)
        
class ExecutorBuilder:
    def __init__(self,use_native=True,label_num=None,grad_clip=None,grad_norm=None,
                 predict_label_num=None,answer_label_num=None,output_label_num=None):
        self.grad_clip = grad_clip
        self.grad_norm = grad_norm
        self.loss = None
        self.optimizer = None
        self.lr_scheduler = None
        self.use_native=use_native
        self._create_optimizer = None
        if self.use_native:
            self.output_label_num = self.predict_label_num = self.answer_label_num = label_num or 3
            self.inference = create_basic_inference(self.output_label_num)
        else:
            self.predict_label_num = predict_label_num or 2
            self.answer_label_num = answer_label_num or 3
            self.inference = seq_ann_inference
        
    def set_optimizer(self,parameters,optim_type,learning_rate=None,
                      reduce_lr_on_plateau=False,**kwargs):
        learning_rate = learning_rate or 1e-3
        def _create_optimizer():
            lr_scheduler = None
            optimizer = optimizer_generator(optim_type,parameters,lr=learning_rate,**kwargs)
            if reduce_lr_on_plateau:
                lr_scheduler = ReduceLROnPlateau(self.optimizer,verbose=True,threshold=0.1)
            return optimizer,lr_scheduler
        self._create_optimizer = _create_optimizer
   
    def set_loss(self,gamma=None,intron_coef=None,other_coef=None):
        if self.use_native:
            loss = FocalLoss(gamma)
        else:    
            loss = SeqAnnLoss(intron_coef=intron_coef,other_coef=other_coef)
        label_loss = LabelLoss(loss)
        label_loss.predict_inference = create_basic_inference(self.predict_label_num)
        label_loss.answer_inference = create_basic_inference(self.answer_label_num)
        self.loss = label_loss
        
    def build(self,executor_weights_path=None):
        exe = BasicExecutor()
        exe.grad_clip = self.grad_clip
        exe.grad_norm = self.grad_norm
        exe.loss = self.loss
        exe.inference = self.inference
        if self._create_optimizer is not None:
            exe.optimizer,exe.lr_scheduler = self._create_optimizer()
        if executor_weights_path is not None:
            weights = torch.load(executor_weights_path)
            exe.load_state_dict(weights)
        return exe
