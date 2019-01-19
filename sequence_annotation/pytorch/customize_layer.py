import torch.nn as nn
import torch.nn.functional as F

class CNN_1D(nn.Module):
    def __init__(self,use_cuda=True,*args,**kwargs):
        super().__init__()
        self._kernel_size = kwargs['kernel_size']
        self.conv = nn.Conv1d(*args,**kwargs)
        if use_cuda:
            self.conv=self.conv.cuda()
    def forward(self,x):
        x = F.pad(x, [0,self._kernel_size-1], 'constant', 0)
        x = self.conv(x)
        return x