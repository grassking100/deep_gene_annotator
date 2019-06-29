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

    def forward(self, output, answer, mask, **kwargs):
        """data shape is N,C,L"""
        if len(output.shape) != 3 or len(answer.shape) != 3:
            raise Exception("Wrong input shape",output.shape,answer.shape)
        if output.shape[0] != answer.shape[0] or output.shape[1] != answer.shape[1]:
            raise Exception("Inconsist batch size or channel size",output.shape,answer.shape)
        N,C,L = output.shape
        #if mask is None:
        #    mask = torch.ones(N,L).cuda()
        mask = mask[:,L].contiguous().view(-1)
        answer = answer[:,:,L].transpose(1,2).reshape(-1,C).max(1)[1]
        output = output.transpose(1,2).reshape(-1,C).log()
        loss_ =  self.loss(output,answer)
        loss_ = loss_*mask
        loss = loss_.sum()/mask.sum()

        return loss

def bce(outputs,answers):
    return BCE(outputs,answers,reduction='none')
    
#Rename it from CodingLoss
class SeqAnnLoss(nn.Module):
    def __init__(self,intron_coef=None,other_coef=None,nonoverlap_coef=None):
        super().__init__()
        self.intron_coef = intron_coef or 1
        self.other_coef = other_coef or 1
        self.nonoverlap_coef = nonoverlap_coef or 1

    def forward(self, output, answer, mask,**kwargs):
        """
            Data shape is N,C,L.
            Output channel order: Transcription, Intron
            Answe channel order: Exon, Intron, Other
        """        
        if len(output.shape) != 3 or len(answer.shape) != 3:
            raise Exception("Wrong input shape",output.shape,answer.shape)
        if output.shape[0] != answer.shape[0] or output.shape[1] != 2 or answer.shape[1] != 3:
            raise Exception("Inconsist batch size or channel size",output.shape,answer.shape)

        N,C,L = output.shape
        #if mask is None:
        #    mask = torch.ones(N,L).cuda()
        answer = answer[:,:,:L].float()
        mask = mask[:,:L].float()
        #Get data    
        #exon_answer = answer[:,0,:]
        intron_answer = answer[:,1,:]
        other_answer = answer[:,2,:]
        transcript_output = output[:,0,:]
        intron_output = output[:,1,:]
        transcript_answer = 1 - other_answer
        other_output = 1-transcript_output
        #Get mask
        transcript_output_mask = (transcript_output >= 0.5).cuda().float()
        overlap_mask = transcript_output_mask*transcript_answer*mask
        non_overlap_mask = (1-overlap_mask)*mask
        #Calculate loss
        other_loss = bce(other_output,other_answer)*self.other_coef*mask
        intron_loss = bce(intron_output,intron_answer)*self.intron_coef*overlap_mask
        non_overlap_loss = bce(intron_output,torch.zeros_like(intron_output))*non_overlap_mask*self.nonoverlap_coef
        loss = other_loss.sum()/(mask.sum())
        loss += intron_loss.sum()/(overlap_mask.sum()+1e-32)
        loss += non_overlap_loss.sum()/(non_overlap_mask.sum()+1e-32)
        return loss
