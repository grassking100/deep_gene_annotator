from abc import abstractmethod,ABCMeta
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy as BCE
import torch.nn.functional as F
from torch.autograd import Variable
from .inference import basic_inference,ann_seq2one_hot_seq
from ..genome_handler.ann_seq_processor import get_start, get_end

EPSILON=1e-32

def mean_by_mask(value,mask):
    return (value * mask).sum()/(mask.sum()+EPSILON)

def bce_loss(outputs,answers,mask=None):
    loss = BCE(outputs,answers,reduction='none')
    if mask is not None:
        loss = mean_by_mask(loss,mask)
    return loss

class ILoss(nn.Module,metaclass=ABCMeta):
    @abstractmethod
    def get_config(self):
        pass

class FocalLoss(ILoss):
    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma or 0

    def get_config(self):
        config = {}
        config['name'] = self.__class__.__name__
        config['gamma'] = self.gamma
        return config
        
    def forward(self, output, answer, mask, **kwargs):
        """data shape is N,C,L"""
        """data shape is N,C,L"""
        if len(output.shape) != 3 or len(answer.shape) != 3:
            raise Exception("Wrong input shape",output.shape,answer.shape)
        if output.shape[0] != answer.shape[0] or output.shape[1] != answer.shape[1]:
            raise Exception("Inconsist batch size or channel size",output.shape,answer.shape)
        _,C,L = output.shape
        mask = mask[:,:L].contiguous().view(-1).float()
        answer = answer[:,:,:L].transpose(1,2).reshape(-1,C).float()
        output = output.transpose(1,2).reshape(-1,C).float()
        loss =  (output+EPSILON).log()
        if self.gamma != 0:
            loss = (1-output)**self.gamma * loss
        loss = (-loss * answer).sum(1)
        loss = mean_by_mask(loss,mask)
        return loss

class CCELoss(nn.Module):
    #Categorical cross entropy
    def __init__(self):
        super().__init__()
        self._loss = FocalLoss(0)

    def get_config(self):
        config = {}
        config['name'] = self.__class__.__name__
        return config
        
    def forward(self, output, answer, mask, **kwargs):
        """data shape is N,C,L"""
        loss =  self._loss(output,answer, mask, **kwargs)
        return loss

class SeqAnnLoss(nn.Module):
    def __init__(self,intron_coef=None,other_coef=None,nontranscript_coef=None,
                 transcript_output_mask=False,transcript_answer_mask=True,
                 mean_by_mask=False):
        super().__init__()
        self.intron_coef = intron_coef or 1
        self.other_coef = other_coef or 1
        self.nontranscript_coef = nontranscript_coef or 0
        self.transcript_output_mask = transcript_output_mask
        self.transcript_answer_mask = transcript_answer_mask
        self.mean_by_mask = mean_by_mask
        
    def get_config(self):
        config = {}
        config['name'] = self.__class__.__name__
        config['intron_coef'] = self.intron_coef
        config['other_coef'] = self.other_coef
        config['nontranscript_coef'] = self.nontranscript_coef
        config['transcript_output_mask'] = self.transcript_output_mask
        config['transcript_answer_mask'] = self.transcript_answer_mask
        config['mean_by_mask'] = self.mean_by_mask
        return config
        
    def forward(self, output, answer, mask,**kwargs):
        """
            Data shape is N,C,L.
            Output channel order: Transcription, Intron|Transcription
            Answer channel order: Exon, Intron, Other
        """
        if len(output.shape) != 3 or len(answer.shape) != 3:
            raise Exception("Wrong input shape",output.shape,answer.shape)
        if output.shape[0] != answer.shape[0]:
            raise Exception("Inconsist batch size",output.shape,answer.shape)
        if output.shape[1] != 2:
            raise Exception("Wrong output channel size, except 2 but got {}".format(output.shape[1]))
        if answer.shape[1] != 3:
            raise Exception("Wrong answer channel size, except 3 but got {}".format(answer.shape[1]))

        _,_,L = output.shape
        answer = answer[:,:,:L].float()
        mask = mask[:,:L].float()
        #Get data
        intron_answer = answer[:,1,:]
        other_answer = answer[:,2,:]
        transcript_output = output[:,0,:]
        intron_transcript_output = output[:,1,:]
        transcript_answer = 1 - other_answer
        other_output = 1-transcript_output
        #Get mask
        transcript_mask = mask
        if self.transcript_output_mask:
            transcript_output_mask = (transcript_output >= 0.5).cuda().float()
            transcript_mask = transcript_mask*transcript_output_mask
        if self.transcript_answer_mask:
            transcript_mask = transcript_mask*transcript_answer
        if self.nontranscript_coef > 0:
            nontranscript_mask = (1-transcript_mask)*mask
        #Calculate loss
        other_loss = bce_loss(other_output,other_answer)*self.other_coef
        intron_loss = bce_loss(intron_transcript_output,intron_answer)*self.intron_coef*transcript_mask

        if self.nontranscript_coef > 0:
            zero = torch.zeros_like(intron_transcript_output)
            nontranscript_loss = bce_loss(intron_transcript_output,zero)*nontranscript_mask*self.nontranscript_coef
        
        other_loss = mean_by_mask(other_loss,mask)
        if self.mean_by_mask:
            intron_loss = mean_by_mask(intron_loss,mask)
        else:
            intron_loss = mean_by_mask(intron_loss,transcript_mask)
        loss = other_loss
        loss = loss + intron_loss
        if self.nontranscript_coef > 0:
            if self.mean_by_mask:
                nontranscript_loss = mean_by_mask(nontranscript_loss,mask)
            else:
                nontranscript_loss = mean_by_mask(nontranscript_loss,nontranscript_mask)
            loss = loss + nontranscript_loss
        loss = loss / (self.intron_coef+self.nontranscript_coef+self.nontranscript_coef)
        return loss

class LabelLoss(nn.Module):
    def __init__(self,loss):
        super().__init__()
        self.loss = loss
        self.predict_inference = basic_inference(3)
        self.answer_inference = basic_inference(3)
        
    def get_config(self):
        config = {}
        config['name'] = self.__class__.__name__
        config['loss_config'] = self.loss.get_config()
        config['predict_inference'] = self.predict_inference.__name__
        config['answer_inference'] = self.answer_inference.__name__
        return config
            
    def forward(self, output, answer, mask ,**kwargs):
        label_predict = self.predict_inference(output,mask)
        label_answer = self.answer_inference(answer,mask)
        loss = self.loss(label_predict,label_answer, mask)
        return loss
