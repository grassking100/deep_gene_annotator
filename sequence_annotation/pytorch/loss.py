import torch.nn as nn
import torch
from torch.nn.functional import binary_cross_entropy as BCE

class CCELoss(nn.Module):
    #Categorical cross entropy
    def __init__(self):
        super().__init__()
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

class SeqAnnLoss(nn.Module):
    def __init__(self,intron_coef=None,other_coef=None,nontranscript_coef=None,
                 transcript_output_mask=False,transcript_answer_mask=True,
                 mean_by_mask=False):
        super().__init__()
        self.intron_coef = 1
        self.other_coef = 1
        self.nontranscript_coef = 0
        if intron_coef is not None:
            self.intron_coef = intron_coef
        if other_coef is not None:
            self.other_coef = other_coef
        if nontranscript_coef is not None:
            self.nontranscript_coef = nontranscript_coef
        self.transcript_output_mask = transcript_output_mask
        self.transcript_answer_mask = transcript_answer_mask
        self.mean_by_mask = mean_by_mask
        
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
        other_loss = bce(other_output,other_answer)*self.other_coef*mask
        intron_loss = bce(intron_transcript_output,intron_answer)*self.intron_coef*transcript_mask

        if self.nontranscript_coef > 0:
            zero = torch.zeros_like(intron_transcript_output)
            nontranscript_loss = bce(intron_transcript_output,zero)*nontranscript_mask*self.nontranscript_coef
        EPSILON=1e-32
        
        other_loss = other_loss.sum()/(mask.sum())
        if not self.mean_by_mask:
            intron_loss = intron_loss.sum()/(transcript_mask.sum()+EPSILON)
        else:
            intron_loss = intron_loss.sum()/(mask.sum())
        loss = other_loss
        loss = loss + intron_loss
        if self.nontranscript_coef > 0:
            if not self.mean_by_mask:
                nontranscript_loss = nontranscript_loss.sum()/(nontranscript_mask.sum()+EPSILON)
            else:
                nontranscript_loss = nontranscript_loss.sum()/(mask.sum())
            loss = loss + nontranscript_loss
        return loss
