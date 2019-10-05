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
    #L = value.shape[1]
    #mask = mask[:,:L].float()
    return (value * mask).sum()/(mask.sum()+EPSILON)

def bce_loss(outputs,answers,mask=None):
    loss = BCE(outputs,answers,reduction='none')
    if mask is not None:
        loss = mean_by_mask(loss,mask)
    return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma or 0

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
        self.loss = FocalLoss(0)

    def forward(self, output, answer, mask, **kwargs):
        """data shape is N,C,L"""
        loss =  self.loss(output,answer, mask, **kwargs)
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
        return loss
    
class SiteLoss(nn.Module):
    def __init__(self,loss):
        super().__init__()
        self.loss = loss
        self.output_inference = basic_inference(3,before=False)
        self.answer_inference = basic_inference(3,before=False)
            
    def forward(self, output, answer, mask ,site_mask,**kwargs):
        site_predict = self.output_inference(output,mask)
        site_answer = self.answer_inference(answer,mask)
        loss = self.loss(site_predict,site_answer, site_mask)
        return loss

class LabelLoss(nn.Module):
    def __init__(self,loss):
        super().__init__()
        self.loss = loss
        self.predict_inference = basic_inference(3)
        self.answer_inference = basic_inference(3)
            
    def forward(self, output, answer, mask ,**kwargs):
        site_predict = self.predict_inference(output,mask)
        site_answer = self.answer_inference(answer,mask)
        loss = self.loss(site_predict,site_answer, mask)
        return loss
    
def signals2site_mask(signals,signal_index):
    N,C,L = signals.shape
    site_mask = np.zeros((N,L))
    signals = np.split(signals,N)
    for item_index,signal in enumerate(signals):
        signal = signal[0]
        signal = ann_seq2one_hot_seq(signal)
        signal = signal.astype(int).astype(str)
        sites = []
        for index in signal_index:
            signal_ = signal[index]
            sites = sites + get_start(''.join(signal_)) + get_end(''.join(signal_))
        for site in set(sites):
            site_mask[item_index,site] = 1
    return site_mask
    
class MixedLoss(nn.Module):
    def __init__(self,label_loss,site_loss,site_mask_method=None):
        super().__init__()
        self.label_loss = label_loss
        self.site_loss = site_loss
        self._site_mask_method = None
        self.site_mask_method = site_mask_method or 'by_answer'
        self.signal_index = [0,1]
        
    @property
    def site_mask_method(self):
        return self._site_mask_method
        
    @site_mask_method.setter
    def site_mask_method(self,value):
        if value in ['by_answer','by_answer_predict','by_mask']:
            self._site_mask_method = value
        else:
            raise Exception("Get weong method {}".format(value))
        
    def forward(self, output, answer, mask, predict_result,**kwargs):
        """
            Data shape is N,C,L.
            Output channel order: *,TSSs,CAs,DSs,ASs
            Answer channel order: *,TSSs,CAs,DSs,ASs
            Predict_result channel order: exon, intron, other
        """
        #Get site_answer with channel order: exon, intron and other
        if self.site_mask_method == 'by_mask':
            site_mask = mask
        else:
            site_answer = self.site_loss.inference(answer)
            site_mask_ = (site_answer.sum(1)>=1).long()
            if self.site_mask_method == 'by_answer':
                site_mask = site_mask_
            else:
                signals = predict_result.detach().cpu().numpy()
                site_mask = signals2site_mask(signals,self.signal_index)
                site_mask = torch.LongTensor(site_mask).cuda()
                site_mask += site_mask_
            site_mask = (site_mask>=1).long()
        
        label_loss = self.label_loss(output, answer, mask)
        site_loss = self.site_loss(output, answer, mask,site_mask=site_mask)
        loss_ = label_loss + site_loss
        return loss_