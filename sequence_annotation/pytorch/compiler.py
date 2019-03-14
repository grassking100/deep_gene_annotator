from ..process.compiler import Compiler
import torch

class SimpleCompiler(Compiler):
    def __init__(self,optimizer,loss,grad_clip=None,grad_norm=None):
        super().__init__()
        if grad_clip is not None and grad_norm is not None:
            raise Exception("Grad_clip and grad_norm cannot be set at the same time")
        self._optimizer_wrapper = optimizer
        self._loss = loss
        self._grad_clip = grad_clip
        self._grad_norm = grad_norm
        self._record['loss_type'] = loss
    def fit(self,model,inputs, labels, lengths,**kwargs):
        model.train(True)
        self._optimizer.zero_grad()
        outputs = model(inputs,lengths=lengths)
        loss_ = self._loss(outputs, labels, lengths=model.saved_lengths, **kwargs)
        loss_.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(),self._grad_clip)
        if self._grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),self._grad_norm)
        self._optimizer.step()
        return loss_.item()
    def evaluate(self,model,inputs, labels,lengths,**kwargs):
        model.train(False)
        with torch.no_grad():
            outputs = model(inputs,lengths=lengths)
            loss_ = self._loss(outputs, labels, lengths=model.saved_lengths,**kwargs)
        return loss_.item()
    def process(self,model):
        self._optimizer = self._optimizer_wrapper(model.parameters())
        self._record['optimizer'] = self._optimizer
    def before_process(self,path=None):
        if path is not None:
            json_path = create_folder(path) + "/setting/compiler.json"
            with open(json_path,'w') as fp:
                json.dump(self._record,fp)