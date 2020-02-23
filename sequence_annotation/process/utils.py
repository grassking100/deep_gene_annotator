from collections import OrderedDict
import math
import numpy as np
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_uniform_

def deep_copy(data):
    if isinstance(data,dict):
        copied = {}
        for key,value in data.items():
            copied[key] = deep_copy(value)
    elif isinstance(data,list):
        copied = []
        for item in data:
            copied.append(deep_copy(item))
    elif isinstance(data,torch.Tensor):
        copied = data.cpu().clone()
    else:
        copied = data
    return copied

def get_copied_state_dict(container):
    weights = OrderedDict()
    for key,data in dict(container.state_dict()).items():
        weights[key] = data.cpu().clone()
    return weights

def get_seq_mask(lengths,max_length=None,to_tensor=True,to_cuda=True):
    max_length = max_length or max(lengths)
    if to_tensor:
        mask = (torch.arange(max_length)[None,:] < torch.LongTensor(lengths)[:,None]).float()
        if to_cuda:
            mask = mask.cuda()    
    else:
        mask = np.zeros((len(lengths),max_length))
        for index,length in enumerate(lengths):
            mask[index,:length] = 1
    return mask

def param_num(model,requires_grad=True):
    if requires_grad:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    return sum([p.numel() for p in model_parameters])

def _get_std_bound(tensor,mode=None,n=None):
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
    std = math.sqrt(1/n)
    return std
    
def xavier_uniform_extend_(tensor, gain=1.,mode=None,n=None):
    #Reference:https://pytorch.org/docs/stable/_modules/torch/nn/init.html
    std_bound = _get_std_bound(tensor,mode=mode,n=n)
    bound = math.sqrt(3.0) * gain * std_bound
    return _no_grad_uniform_(tensor, -bound, bound)

def get_name_parameter(model,names):
    parameters = []
    returned_names = []
    for name_,parameter in model.named_parameters():
        for target_name in names:
            if target_name in name_:
                parameters.append(parameter)
                returned_names.append(name_)
    return returned_names,parameters
