import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.init import ones_,zeros_,uniform_,normal_,constant_

class CNN_1D(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self._kernel_size = kwargs['kernel_size']
        self.cnn = nn.Conv1d(*args,**kwargs).cuda()
        self.reset_parameters()
    def reset_parameters(self):
        normal_(self.cnn.weight,0,1)
        constant_(self.cnn.bias,0.5)
    def forward(self,x):
        x = F.pad(x, [0,self._kernel_size-1], 'constant', 0)
        x = self.cnn(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, class_num, gamma=0,ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self._ignore_index = ignore_index
        self._class_num = class_num
    def forward(self, pt, target):
        target = target.float()
        pt = pt.contiguous()
        pt = pt.view(-1, self._class_num)
        target = torch.transpose(target, 1, 2).contiguous() 
        target = target.view(-1, self._class_num)
        if self._ignore_index is not None:
            mask = (target.max(1)[0] != self._ignore_index).float()
        #alpha = []
        #for index in range(self._class_num):
        #    alpha.append((target==index).sum())
        #alpha = torch.FloatTensor(alpha).cuda()
        #at = alpha.gather(0,target.data.view(-1))
        #loss_func = torch.nn.NLLLoss(ignore_index=self._ignore_index,reduction ='none',size_average=False)
        decay_cooef = (1-pt)**self.gamma
        loss_ =  -decay_cooef* (pt+1e-10).log() * target
        if self._ignore_index is not None:
            loss_ = loss_.sum(1)*mask
            loss = loss_.sum()/mask.sum()
        else:
            loss = loss_.mean()
        return loss

def init_GRU(gru):
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)