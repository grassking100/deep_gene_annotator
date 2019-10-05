import torch.nn as nn
import torch
from abc import abstractmethod
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
        self.inference = basic_inference(3)

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
            outputs,lengths,masks = _predict(model,inputs,lengths,inference=self.inference,**kwargs)
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
        model.train()
        label_model, discrim_model = model.gan,model.discrim
        #train generator
        self._label_optimizer.zero_grad()
        predict_labels,lengths_ = label_model(inputs,lengths=lengths)
        masks = get_seq_mask(lengths_)
        outputs = self.inference(predict_labels,masks)
        label_status = discrim_model(inputs,predict_labels,lengths=lengths_)
        ONES = torch.ones(*label_status.shape).cuda()
        ZEROS = torch.zeros(*label_status.shape).cuda()
        gan_label_loss = bce_loss(label_status,ONES,masks)
        class_loss = self.loss(predict_labels, labels, masks) 
        gan_loss = gan_label_loss*0.01 + class_loss
        gan_loss.backward()
        self._label_optimizer.step()
        #train discriminator
        self._discrim_optimizer.zero_grad()
        predict_labels,lengths_ = label_model(inputs,lengths=lengths)
        masks = get_seq_mask(lengths_)
        if self.reverse_inference is not None:
            reverse_inference_labels = self.reverse_inference(labels,masks)
        else:
            reverse_inference_labels = labels
        
        real_status = discrim_model(inputs,reverse_inference_labels,lengths=lengths_)
        label_status = discrim_model(inputs,predict_labels,lengths=lengths_)
        disrim_label_loss = bce_loss(label_status,ZEROS,masks)
        disrim_real_loss = bce_loss(real_status,ONES,masks)
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
