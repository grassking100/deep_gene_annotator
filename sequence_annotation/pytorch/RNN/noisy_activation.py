import math
from torch import nn
from torch.autograd import Function
import torch
import noisy_sigmoid_cuda
from torch.nn.init import ones_,zeros_,uniform_,normal_, constant_
from torch.nn import Hardtanh, Sigmoid,Tanh,ReLU
class NoisyHardSigmoidFunction(Function):
    @staticmethod
    def forward(ctx,x,alpha,c,p,alpha_complement,alpha_complement_sgn,random):
        outputs = noisy_sigmoid_cuda.forward(x,alpha,c,p,alpha_complement,alpha_complement_sgn,random)
        variables = outputs + [x,alpha,c,p,alpha_complement,alpha_complement_sgn,random]
        ctx.save_for_backward(*variables)
        return outputs

    @staticmethod
    def backward(ctx, d_y):
        saved_variables = []
        for var in ctx.saved_variables:
            saved_variables.append(var.contiguous())
        d_o = noisy_sigmoid_cuda.backward(d_y.contiguous(),*saved_variables)
        return d_o        

class NoisyHardSigmoid(nn.Module):
    def __init__(self,alpha=None,c=None,p=None):
        super().__init__()
        self._alpha = alpha or 1
        self._c = c or 0.05
        self._p = p or 1
        self._alpha_complement = 1-self._alpha
        if self._alpha_complement>0:
            self._alpha_complement_sgn = 1
        else:
            self._alpha_complement_sgn = -1

    def forward(self, x):

        if self.training:
            random = torch.abs(torch.randn_like(x))
            return NoisyHardSigmoidFunction.apply(x,self._alpha,self._c,self._p,
                                                  self._alpha_complement,self._alpha_complement_sgn,random)
        else:
            return torch.sigmoid(x)
