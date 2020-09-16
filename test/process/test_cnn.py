import numpy as np
import torch
import unittest
from sequence_annotation.process.cnn import Conv1d


class TestCNN(unittest.TestCase):
    
    def test_same_cnn(self):
        x = torch.FloatTensor([[[1, 1, 1, 1, 0]]]).cuda()
        cnn = Conv1d(1,1,3)
        cnn._cnn.weight = torch.nn.Parameter(torch.ones(1, 1, 3))
        cnn._cnn.bias = torch.nn.Parameter(torch.zeros(1))
        cnn = cnn.cuda()
        y = cnn(x, lengths=[4])[0].int().detach().cpu().numpy()[:, :, :-1]
        real = np.array([[[2, 3, 3, 2]]], dtype='int32')
        np.testing.assert_equal(real, y)

    def test_stack_same_cnn(self):
        x = torch.FloatTensor([[[1, 1, 1, 1, 0]]]).cuda()
        cnn = Conv1d(1,1,3)
        cnn._cnn.weight = torch.nn.Parameter(torch.ones(1, 1, 3))
        cnn._cnn.bias = torch.nn.Parameter(torch.zeros(1))
        cnn = cnn.cuda()
        y, l, m = cnn(x, lengths=[4])
        z = cnn(y, lengths=l)[0].int().detach().cpu().numpy()[:, :, :-1]
        real = np.array([[[5, 8, 8, 5]]], dtype='int32')
        np.testing.assert_equal(real, z)

