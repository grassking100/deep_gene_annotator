import numpy as np
import torch

def get_seq_mask(lengths,max_lengths=None):
    max_lengths = max_lengths or max(lengths)
    mask = np.zeros((len(lengths),max_lengths))
    for index,length in enumerate(lengths):
        mask[index,:length]=1
    mask = torch.FloatTensor(mask).cuda()    
    return mask
