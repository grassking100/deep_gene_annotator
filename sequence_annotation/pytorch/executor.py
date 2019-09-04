import torch.nn as nn
import torch
from abc import abstractmethod
from .loss import CCELoss
from .utils import get_seq_mask

bce_loss = nn.BCELoss(reduction='none')

def mean_by_mask(value,mask):
    L = value.shape[1]
    mask = mask[:,:L].float()
    return (value * mask).sum()/(mask.sum())

def _inference(outputs,masks,inference=None):
    if inference is not None:
        outputs = inference(outputs,masks)
    else:
        outputs = outputs.transpose(0,1)
        if masks is not None:
            L = outputs.shape[2]
            outputs = outputs*(masks[:,:L].float())
        outputs = outputs.transpose(0,1)
    return outputs

def _evaluate(loss,model,inputs, labels,lengths,inference=None):
    model.train(False)
    with torch.no_grad():
        outputs,lengths = model(inputs,lengths=lengths,return_length=True)
        masks = get_seq_mask(lengths)
        outputs = outputs.float()
        loss_ = loss(outputs, labels, masks).item()
        outputs = _inference(outputs,masks,inference)
    return loss_,outputs,lengths,masks

def _predict(model,inputs,lengths,inference=None):
    model.train(False)
    with torch.no_grad():
        outputs,lengths = model(inputs,lengths=lengths,return_length=True)
        masks = get_seq_mask(lengths)
        outputs = outputs.float()
        outputs = _inference(outputs,masks,inference)
    return outputs,lengths,masks

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
            outputs,lengths,masks = self.predict(model,inputs,lengths,**kwargs)
            return {},outputs,lengths,masks

class BasicExecutor(_Executor):
    def __init__(self):
        super().__init__()
        self.grad_clip = None
        self.grad_norm = None
        self.optimizer = None

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['grad_clip'] = self.grad_clip
        config['grad_norm'] = self.grad_norm
        config['optimizer'] = self.optimizer.state_dict()

    def fit(self,model,inputs, labels, lengths, **kwargs):
        if self.optimizer is None:
            raise Exception("Exectutor must set optimizer for fitting")
        model.train(True)
        self.optimizer.zero_grad()
        outputs,lengths = model(inputs,lengths=lengths,return_length=True)
        outputs = outputs.float()
        masks = get_seq_mask(lengths)
        loss_ = self.loss(outputs, labels, masks, **kwargs)
        loss_.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_value_(model.parameters(),self.grad_clip)
        if self.grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(),self.grad_norm)
        self.optimizer.step()
        outputs = _inference(outputs, masks, self.inference)
        return {'loss':loss_.item()},outputs,lengths,masks

    def process(self,model):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters())

class GANExecutor(_Executor):
    def __init__(self):
        super().__init__()
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
        #train generator
        self._label_optimizer.zero_grad()
        predict_labels,lengths_ = label_model(inputs,lengths=lengths,return_length=True)
        masks = get_seq_mask(lengths_)
        outputs = _inference(predict_labels,masks,self.inference)
        label_status = discrim_model(inputs,predict_labels,lengths=lengths_)
        
        gan_label_loss = bce_loss(label_status,torch.ones(*label_status.shape).cuda())
        gan_label_loss = mean_by_mask(gan_label_loss,masks)
        class_loss = self.loss(predict_labels, labels, masks) 
        gan_loss = gan_label_loss + class_loss
        gan_loss.backward()
        self._label_optimizer.step()
        #train discriminator
        self._discrim_optimizer.zero_grad()
        predict_labels,lengths_ = label_model(inputs,lengths=lengths,return_length=True)
        masks = get_seq_mask(lengths_)
        if self.reverse_inference is not None:
            reverse_inference_labels = self.reverse_inference(labels,masks)
        else:
            reverse_inference_labels = labels
        
        real_status = discrim_model(inputs,reverse_inference_labels,lengths=lengths_)
        label_status = discrim_model(inputs,predict_labels,lengths=lengths_)
        disrim_label_loss = bce_loss(label_status,torch.zeros(*label_status.shape).cuda())
        disrim_label_loss = mean_by_mask(disrim_label_loss,masks)
        disrim_real_loss = bce_loss(real_status,torch.ones(*real_status.shape).cuda())
        disrim_real_loss = mean_by_mask(disrim_real_loss,masks)
        disrim_loss = disrim_label_loss + disrim_real_loss
        disrim_loss.backward()
        self._discrim_optimizer.step()
        real_status_mean = mean_by_mask(real_status,masks)
        label_status_mean = mean_by_mask(label_status,masks)
        return {'loss':gan_loss.item(),
                'gan_label_loss':gan_label_loss.item(),
                'class_loss':class_loss.item(),
                'disrim_loss':disrim_loss.item(),
                'disrim_real_loss':disrim_real_loss.item(),
                'disrim_label_loss':disrim_label_loss.item(),
                'real_conf_mean':real_status_mean.item(),
                'label_conf_mean':label_status_mean.item()},outputs,lengths_,masks

    def evaluate(self,model,inputs,labels,lengths,**kwargs):
        return super().evaluate(model.gan,inputs,labels,lengths=lengths,**kwargs)

    def predict(self,model,inputs,lengths,**kwargs):
        return super().predict(model.gan,inputs,lengths=lengths,**kwargs)

    def process(self,model):
        if self.optimizer is None:
            self._label_optimizer = torch.optim.Adam(model.gan.parameters())
            self._discrim_optimizer = torch.optim.Adam(model.discrim.parameters())
        else:
            self._label_optimizer, self._discrim_optimizer = self.optimizer
