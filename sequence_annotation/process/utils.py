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

def param_num(model):
    return sum([p.numel() for p in model.parameters()])

def xavier_uniform_in_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1 / float(fan_in))
    bound = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -bound, bound)
