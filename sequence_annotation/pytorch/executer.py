from .loss import CCELoss
import torch.nn as nn
import torch

bce_loss = nn.BCELoss(reduction='mean')

def _evaluate(loss,model,inputs, labels,lengths,inference=None,**kwargs):
    model.train(False)
    with torch.no_grad():
        outputs = model(inputs,lengths=lengths)
        loss_ = loss(outputs, labels,**kwargs)
        if inference is not None:
            outputs = inference(outputs,labels)
    return loss_.item(),outputs,model.saved_lengths,model.named_parameters()

def _predict(model,inputs,lengths,inference=None,**kwargs):
    model.train(False)
    with torch.no_grad():
        outputs = model(inputs,lengths=lengths)
        if inference is not None:
            outputs = inference(outputs)
    return outputs

class ModelExecutor:
    def __init__(self):
        self.loss = CCELoss()
        self.inference = None
        self.grad_clip = None
        self.grad_norm = None
        self.optimizer = None
        
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
            outputs = self.inference(outputs,labels)
        return loss_.item(),outputs,model.saved_lengths,model.named_parameters()

    def evaluate(self,model,inputs, labels,lengths,**kwargs):
        return _evaluate(self.loss,model,inputs, labels,lengths,self.inference)
    
    def predict(self,model,inputs,lengths,**kwargs):
        return _predict(model,inputs,lengths,self.inference)

    def process(self,model):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters())

class GANExecutor():
    def __init__(self):
        self.loss = CCELoss()
        self.inference = None
        self.reverse_inference = None
        self.optimizer = None
        self._label_optimizer = None
        self._discrim_optimizer = None
        
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
        loss_ = self.loss(outputs, labels, **kwargs) + bce_loss(outputs_status,torch.ones(len(outputs_status)).cuda())
        loss_.backward()
        if self.inference is not None:
            outputs = self.inference(outputs,labels)
        return loss_.item(),outputs,label_model.saved_lengths,label_model.named_parameters()

    def evaluate(self,model,inputs,labels,lengths,**kwargs):
        return _evaluate(self.loss,model.gan,inputs,labels,lengths,self.inference)
    
    def predict(self,model,inputs,lengths,**kwargs):
        return _predict(model.gan,inputs,lengths,self.inference)
        
    def process(self,model):
        if self.optimizer is None:
            self._label_optimizer = torch.optim.Adam(model.gan.parameters())
            self._discrim_optimizer = torch.optim.Adam(model.discrim.parameters())
        else:
            self._label_optimizer, self._discrim_optimizer = self.optimizer
