import torch.nn as nn
import torch
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
        _,C,L = output.shape
        mask = mask[:,:L].contiguous().view(-1).float()
        answer = answer[:,:,:L].transpose(1,2).reshape(-1,C).max(1)[1]
        output = output.transpose(1,2).reshape(-1,C).log().float()
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
            Output channel order: Transcription, Intron|Transcription
            Answer channel order: Exon, Intron, Other
        """
        if len(output.shape) != 3 or len(answer.shape) != 3:
            raise Exception("Wrong input shape",output.shape,answer.shape)
        if output.shape[0] != answer.shape[0] or output.shape[1] != 2 or answer.shape[1] != 3:
            raise Exception("Inconsist batch size or channel size",output.shape,answer.shape)

        _,_,L = output.shape
        answer = answer[:,:,:L].float()
        mask = mask[:,:L].float()
        #Get data
        #exon_answer = answer[:,0,:]
        intron_answer = answer[:,1,:]
        other_answer = answer[:,2,:]
        transcript_output = output[:,0,:]
        intron_transcript_output = output[:,1,:]
        transcript_answer = 1 - other_answer
        other_output = 1-transcript_output
        #Get mask
        transcript_output_mask = (transcript_output >= 0.5).cuda().float()
        overlap_mask = transcript_output_mask*transcript_answer*mask
        non_overlap_mask = (1-overlap_mask)*mask
        #Calculate loss
        other_loss = bce(other_output,other_answer)*self.other_coef*mask
        intron_loss = bce(intron_transcript_output,intron_answer)
        intron_loss = intron_loss*self.intron_coef*overlap_mask
        non_overlap_loss = bce(intron_transcript_output,torch.zeros_like(intron_transcript_output))
        non_overlap_loss = non_overlap_loss*non_overlap_mask*self.nonoverlap_coef
        loss = other_loss.sum()/(mask.sum())
        loss += intron_loss.sum()/(overlap_mask.sum()+1e-32)
        loss += non_overlap_loss.sum()/(non_overlap_mask.sum()+1e-32)
        return loss

class SeqAnnAltLoss(nn.Module):
    def __init__(self,intron_coef=None,other_coef=None,alt_coef=None):
        super().__init__()
        self.intron_coef = intron_coef or 1
        self.other_coef = other_coef or 1
        self.alt_coef = alt_coef or 1

    def forward(self, output, answer, mask,**kwargs):
        """
            Data shape is N,C,L.
            Output channel order: Transcription, Intron|Transcription, Alternative Intron|Intron
            Answer channel order: Alternative Intron, Exon, Nonalternative Intron, Other
        """
        if len(output.shape) != 3 or len(answer.shape) != 3:
            raise Exception("Wrong input shape",output.shape,answer.shape)
        if output.shape[0] != answer.shape[0] or output.shape[1] != 3 or answer.shape[1] != 4:
            raise Exception("Inconsist batch size or channel size",output.shape,answer.shape)
        N,C,L = output.shape
        answer = answer[:,:,:L].float()
        mask = mask[:,:L].float()
        #Get data
        alt_intron_answer = answer[:,0,:]
        nonalt_intron_answer = answer[:,2,:]
        other_answer = answer[:,3,:]
        intron_answer = alt_intron_answer+nonalt_intron_answer
        other_answer = answer[:,3,:]
        transcript_output = output[:,0,:]
        intron_trans_output = output[:,1,:]
        alt_intron_output = output[:,2,:]
        transcript_answer = 1 - other_answer
        other_output = 1 - transcript_output
        #Get mask
        transcript_overlap_mask = transcript_answer*mask#*(transcript_output >= 0.5).cuda().float()
        intron_overlap_mask = intron_answer*mask#*((intron_trans_output*transcript_output) >= 0.5).cuda().float()
        #Calculate loss
        other_loss = bce(other_output,other_answer)*self.other_coef*mask
        intron_loss = bce(intron_trans_output,intron_answer)*self.intron_coef*transcript_overlap_mask
        alt_loss = bce(alt_intron_output,alt_intron_answer)*self.alt_coef*intron_overlap_mask
        loss = other_loss.sum()/(mask.sum())
        loss += intron_loss.sum()/(transcript_overlap_mask.sum()+1e-32)
        loss += alt_loss.sum()/(intron_overlap_mask.sum()+1e-32)
        return loss
