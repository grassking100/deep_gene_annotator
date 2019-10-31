from collections import OrderedDict
import math
import numpy as np
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_uniform_,constant_

def get_copied_state_dict(model):
    weights = OrderedDict()
    for key,tensor in dict(model.state_dict()).items():
        weights[key] = tensor.clone()
    return weights

def get_seq_mask(lengths,max_lengths=None,to_tensor=True,to_cuda=True):
    max_lengths = max_lengths or max(lengths)
    mask = np.zeros((len(lengths),max_lengths))
    for index,length in enumerate(lengths):
        mask[index,:length] = 1
    if to_tensor:
        mask = torch.FloatTensor(mask)
        if to_cuda:
            mask = mask.cuda()    
    return mask

def param_num(model,requires_grad=True):
    if requires_grad:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    return sum([p.numel() for p in model_parameters])

def xavier_uniform_extend_(tensor, gain=1.,mode=None,n=None):
    #Reference:https://pytorch.org/docs/stable/_modules/torch/nn/init.html
    mode = mode or 'both'
    VALID_MODES = ['fan_in','fan_out','both']
    if mode not in VALID_MODES:
        raise Exception("Got wrong mode {}, expect {}".format(mode,VALID_MODES))
    if n is None:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        if mode == 'both':
            n = float(fan_in + fan_out)/2
        elif mode == 'fan_in':
            n = fan_in
        else:
            n = fan_out
    std = gain * math.sqrt(1/n)
    bound = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -bound, bound)
