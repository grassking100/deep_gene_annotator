import os
import sys
import torch
import random
import deepdish as dd
import numpy as np

def backend_deterministic(deterministic,benchmark=False):
    if deterministic:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
    torch.backends.cudnn.enabled = not deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
