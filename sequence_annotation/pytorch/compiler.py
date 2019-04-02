from ..process.compiler import Compiler
from .customize_layer import SeqAnnLoss
import torch
import json

class SeqAnnCompiler(Compiler):
    def __init__(self):
        self._settings = {}
        self.loss = SeqAnnLoss()
        self.optimizer_settings = {}
        self.grad_clip = None
        self.grad_norm = None
        self.optimizer_class = torch.optim.Adam

    def fit(self,model,inputs, labels, lengths,**kwargs):
        model.train(True)
        self._optimizer.zero_grad()
        outputs = model(inputs,lengths=lengths)
        loss_ = self.loss(outputs, labels, lengths=model.saved_lengths, **kwargs)
        loss_.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(),self.grad_clip)
        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),self.grad_norm)
        self._optimizer.step()
        return loss_.item()

    def evaluate(self,model,inputs, labels,lengths,**kwargs):
        model.train(False)
        with torch.no_grad():
            outputs = model(inputs,lengths=lengths)
            loss_ = self.loss(outputs, labels, lengths=model.saved_lengths,**kwargs)
        return loss_.item()

    def process(self,model):
        if hasattr(self.loss,'transitions'):
            self.loss.transitions = model.CRF.transitions
        if self.optimizer_class is not None:
            self._optimizer = self.optimizer_class(model.parameters(),**self.optimizer_settings)

    def before_process(self,path=None):
        if path is not None:
            json_path = path + "/compiler_setting.json"
            for key in ['optimizer_class','optimizer_settings','grad_clip','grad_norm']:
                self._settings[key] = getattr(self,key)
            with open(json_path,'w') as fp:
                try:
                    json.dump(self._settings,fp)
                except TypeError:
                     fp.write(str(self._settings))
