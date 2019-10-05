import numpy as np
import torch

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
