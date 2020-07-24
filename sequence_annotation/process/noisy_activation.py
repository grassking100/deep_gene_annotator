from abc import abstractmethod
import torch
from torch.nn import Hardtanh, ReLU
from torch import nn

import torch
import torch.nn.functional as F
from torch import nn

def sgn(x):
    return (x > 0).float() * 2 - 1


class SimpleHalfNoisyReLU(nn.Module):
    def forward(self, x):
        h = F.relu(x)
        sigma = (torch.sigmoid(h - x) - 0.5)**2
        if self.training:
            random = torch.abs(torch.randn_like(x))
            return h + sigma * random
        else:
            #mean of half normal distribution with std=1 is sqrt(2/pi)=0.798
            return h + sigma * 0.798
        
class SimpleNoisyReLU(nn.Module):
    def forward(self, x):
        h = F.relu(x)
        if self.training:
            sigma = (torch.sigmoid(h - x) - 0.5)**2
            random = torch.randn_like(x)
            return h + sigma * random
        else:
            return h

class SymHardSigmoid:
    def __init__(self):
        self.act = Hardtanh(min_val=-0.5, max_val=0.5)

    def __call__(self, x):
        return self.act(x) + 0.5


class NoisyHardAct(nn.Module):
    def __init__(self, alpha=None, c=None, p=None):
        super().__init__()
        self._alpha = 1 if alpha is None else alpha
        self._c = 0.05 if c is None else c
        self._p = 1 if p is None else p
        self._hard_function = self._get_hard_function()
        self._alpha_complement = 1 - self._alpha
        if self._alpha_complement > 0:
            self._sgn = 1
        else:
            self._sgn = -1

    @abstractmethod
    def _get_hard_function(self):
        pass

    def forward(self, x):
        h = self._hard_function(x)
        if self.training:
            if self._alpha == 1:
                native_result = h
            else:
                native_result = self._alpha * h + self._alpha_complement * x
            random = torch.abs(torch.randn_like(x))
            diff = h - x
            d = (-sgn(x) * self._sgn).to(x.dtype)
            c_d_random = self._c * d * random
            if self._p != 1:
                diff = self._p * diff
            return native_result + c_d_random * (torch.sigmoid(diff) - 0.5)**2
        else:
            return h


class NoisyHardTanh(NoisyHardAct):
    def _get_hard_function(self):
        return Hardtanh()


class NoisyHardSigmoid(NoisyHardAct):
    def _get_hard_function(self):
        return hard_sigmoid


class SymNoisyHardSigmoid(NoisyHardAct):
    def _get_hard_function(self):
        return SymHardSigmoid()


class NoisyReLU(NoisyHardAct):
    def _get_hard_function(self):
        return ReLU()
