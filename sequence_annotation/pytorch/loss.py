import torch.nn as nn
import torch
from torch import functional as F
from torch.nn.functional import binary_cross_entropy as BCE

class CCELoss(nn.Module):
    #Categorical cross entropy
    def __init__(self):
        super().__init__()
        self.ignore_value = -1
        self.loss = nn.NLLLoss(reduction='none')

    def forward(self, output, answer,**kwargs):
        """data shape is N,C,L"""
        if len(output.shape) != 3 or len(answer.shape) != 3:
            raise Exception("Wrong input shape",output.shape,answer.shape)
        if output.shape[0] != answer.shape[0] or output.shape[1] != answer.shape[1]:
            raise Exception("Inconsist batch size or channel size",output.shape,answer.shape)
        C = output.shape[1]
        input_length = output.shape[2]
        if list(output.shape) != list(answer.shape):
            answer = answer.transpose(0,2)[:input_length].transpose(0,2)
        if self.ignore_value is not None:
            mask = (answer.max(1)[0] != self.ignore_value).float()
        output = output.transpose(1,2).reshape(-1,C).log()
        answer = answer.transpose(1,2).reshape(-1,C)
        answer = answer.max(1)[1]
        mask = mask.contiguous().view(-1)
        loss_ =  self.loss(output,answer)
        if self.ignore_value is not None:
            loss_ = loss_*mask
            loss = loss_.sum()/mask.sum()
        else:
            loss = loss_.mean()
        return loss

#Rename it from CodingLoss
class SeqAnnLoss(nn.Module):
    def __init__(self,exon_coef=None,other_coef=None,nonoverlap_coef=None):#intron_coef=None
        super().__init__()
        self.ignore_value = -1
        self.exon_coef = exon_coef or 1
        #Remove intron coef,because intron_loss will be removed
        #self.intron_coef = intron_coef or 1
        self.other_coef = other_coef or 2
        self.nonoverlap_coef = nonoverlap_coef or 0.5

    def forward(self, output, answer,**kwargs):
        """Data shape is N,C,L.Output channel order:Transcrioutput,Exon"""
        if len(output.shape) != 3 or len(answer.shape) != 3:
            raise Exception("Wrong input shape",output.shape,answer.shape)
        if output.shape[0] != answer.shape[0] or output.shape[1] != 2 or answer.shape[1] != 3:
            raise Exception("Inconsist batch size or channel size",output.shape,answer.shape)
        input_length = output.shape[2]
        if list(output.shape) != list(answer.shape):
            answer = answer.transpose(0,2)[:input_length].transpose(0,2)
        answer = answer.float()
        if self.ignore_value is not None:
            mask = (answer.max(1)[0] != self.ignore_value).float()
        other_answer = answer[:,2,:]
        transcrioutput_answer = 1-other_answer
        transcrioutput_answer_mask = (transcrioutput_answer >= 0.5).float()
        transcrioutput_output = output[:,0,:]
        other_output = 1-transcrioutput_output
        transcrioutput_output_mask = (transcrioutput_output >= 0.5).float()
        exon_answer = answer[:,0,:]
        #intron_answer = answer[:,1,:]
        exon_output = output[:,1,:]
        #intron_output = (1-exon_output)
        overlap_mask = transcrioutput_output_mask*transcrioutput_answer_mask*mask
        non_overlap_mask = (1-overlap_mask)*mask
        other_loss = BCE(other_output,other_answer,reduction='none')*self.other_coef*mask
        exon_loss = BCE(exon_output,exon_answer,reduction='none')*self.exon_coef*overlap_mask
        #Remove intron_loss will be removed, becuase it is equal exon_loss
        #intron_loss = BCE(intron_output,intron_answer,reduction='none')*self.intron_coef*overlap_mask
        non_overlap_loss = BCE(exon_output,torch.ones_like(exon_output))*non_overlap_mask*self.nonoverlap_coef
        if self.ignore_value is not None:
            loss = other_loss.sum()/(mask.sum())
            loss += exon_loss.sum()/(overlap_mask.sum()+1e-32)
            #loss += (intron_loss+exon_loss).sum()/(overlap_mask.sum()+1e-32)
            loss += non_overlap_loss.sum()/(non_overlap_mask.sum()+1e-32)
        else:
            #loss = (other_loss+intron_loss+exon_loss+non_overlap_loss).mean()
            loss = (other_loss+exon_loss+non_overlap_loss).mean()
        return loss
