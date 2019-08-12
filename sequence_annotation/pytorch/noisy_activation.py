from abc import abstractmethod
import torch
from torch import randn
from torch.nn import Hardtanh, Sigmoid,Tanh,ReLU
from torch import nn

hard_sigmoid = Hardtanh(min_val=0)

def sgn(x):
    return (x>0).float()*2-1

class NoisyHardAct(nn.Module):
    def __init__(self,alpha=None,c=None,p=None):
        super().__init__()
        self._alpha = alpha or 1
        self._c = c or 0.05
        self._p = p or 1
        self._hard_function = self._get_hard_function()
        self._alpha_complement = 1-self._alpha
        if self._alpha_complement > 0:
            self._sgn = 1
        else:
            self._sgn = -1

    @abstractmethod
    def _get_hard_function(self):
        pass

    def forward(self,x):
        h = self._hard_function(x)
        if self.training:
            random = torch.abs(torch.randn_like(x))
            diff = h-x
            d = -sgn(x)*self._sgn
            if self._alpha == 1:
                native_result = h
            else:
                native_result = self._alpha*h+self._alpha_complement*x
            if self._p == 1:
                diff = self._p*diff
            sigma = self._c*(torch.sigmoid(diff)-0.5)**2
            return native_result+(d*sigma*random)
        else:
            return h

class NoisyHardTanh(NoisyHardAct):
    def _get_hard_function(self):
        return Hardtanh()

class NoisyHardSigmoid(NoisyHardAct):
    def _get_hard_function(self):
        return hard_sigmoid

class NoisyReLU(NoisyHardAct):
    def _get_hard_function(self):
        return ReLU()