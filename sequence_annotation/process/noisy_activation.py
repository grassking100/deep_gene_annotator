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
