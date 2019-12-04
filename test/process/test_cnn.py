import numpy as np
import torch
import unittest
from sequence_annotation.process.cnn import Conv1d

class TestCNN(unittest.TestCase):
    def test_partial_cnn(self):
        x=torch.FloatTensor([[[1,1,1,1,0]]]).cuda()
        cnn = Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding_handle='partial')
        cnn.weight = torch.nn.Parameter(torch.ones(1,1,3))
        cnn.bias = torch.nn.Parameter(torch.zeros(1))
        cnn = cnn.cuda()
        y=cnn(x,lengths=[4])[0].int().detach().cpu().numpy()[:,:,:-1]
        real=np.array([[[3,3,3,3]]],dtype='int32')
        np.testing.assert_equal(real,y)

    def test_stack_partial_cnn(self):
        x=torch.FloatTensor([[[1,1,1,1,0]]]).cuda()
        cnn = Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding_handle='partial')
        cnn.weight = torch.nn.Parameter(torch.ones(1,1,3))
        cnn.bias = torch.nn.Parameter(torch.zeros(1))
        cnn = cnn.cuda()
        y,l,w,m=cnn(x,lengths=[4])
        z=cnn(y,lengths=l,weights=w)[0].int().detach().cpu().numpy()[:,:,:-1]
        real=np.array([[[9,9,9,9]]],dtype='int32')
        np.testing.assert_equal(real,z)
        
    def test_same_cnn(self):
        x=torch.FloatTensor([[[1,1,1,1,0]]]).cuda()
        cnn = Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding_handle='same')
        cnn.weight = torch.nn.Parameter(torch.ones(1,1,3))
        cnn.bias = torch.nn.Parameter(torch.zeros(1))
        cnn = cnn.cuda()
        y=cnn(x,lengths=[4])[0].int().detach().cpu().numpy()[:,:,:-1]
        real=np.array([[[2,3,3,2]]],dtype='int32')
        np.testing.assert_equal(real,y)
        
    def test_stack_same_cnn(self):
        x=torch.FloatTensor([[[1,1,1,1,0]]]).cuda()
        cnn = Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding_handle='same')
        cnn.weight = torch.nn.Parameter(torch.ones(1,1,3))
        cnn.bias = torch.nn.Parameter(torch.zeros(1))
        cnn = cnn.cuda()
        y,l,w,m=cnn(x,lengths=[4])
        z=cnn(y,lengths=l,weights=w)[0].int().detach().cpu().numpy()[:,:,:-1]
        real=np.array([[[5,8,8,5]]],dtype='int32')
        np.testing.assert_equal(real,z)
        
    def test_valid_cnn(self):
        x=torch.FloatTensor([[[1,1,1,1,1,1,0]],[[1,1,1,1,1,0,0]]]).cuda()
        cnn = Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding_handle='valid')
        cnn.weight = torch.nn.Parameter(torch.ones(1,1,3))
        cnn.bias = torch.nn.Parameter(torch.zeros(1))
        cnn = cnn.cuda()
        y=cnn(x,lengths=[6,5])[0].int().detach().cpu().cpu().numpy()
        real=np.array([[[3,3,3,3]],[[3,3,3,0]]],dtype='int32')
        np.testing.assert_equal(real,y)
        
    def test_stack_valid_cnn(self):
        x=torch.FloatTensor([[[1,1,1,1,1,1,0]],[[1,1,1,1,1,0,0]]]).cuda()
        cnn = Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding_handle='valid')
        cnn.weight = torch.nn.Parameter(torch.ones(1,1,3))
        cnn.bias = torch.nn.Parameter(torch.zeros(1))
        cnn = cnn.cuda()
        y,l,w,m=cnn(x,lengths=[6,5])
        z=cnn(y,lengths=l,weights=w)[0].int().detach().cpu().numpy()
        real=np.array([[[9,9]],[[9,0]]],dtype='int32')
        np.testing.assert_equal(real,z)
