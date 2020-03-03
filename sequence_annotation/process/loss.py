from abc import abstractmethod,ABCMeta,abstractproperty
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy as BCE
from .inference import create_basic_inference

EPSILON=1e-32

def sum_by_mask(value,mask):
    return (value * mask).sum()

def mean_by_mask(value,mask):
    return sum_by_mask(value,mask)/(mask.sum()+EPSILON)

def bce_loss(output,answer,mask=None,return_mean=True):
    #N,L
    if len(output.shape) != 2 or len(answer.shape) != 2:
        raise Exception("Wrong shape")

    if output.shape != answer.shape:
        raise Exception("Inconsist batch size or length size",output.shape,answer.shape)
        
    if mask is not None and mask.shape != output.shape:
        raise Exception("Wrong shape")

    loss = BCE(output,answer,reduction='none')
    if return_mean:
        if mask is not None:
            loss = mean_by_mask(loss,mask)
        else:
            loss = loss.sum() / loss.shape[1]
    return loss

class ILoss(nn.Module,metaclass=ABCMeta):
    @abstractproperty
    def accumulated_data(self):
        pass
    @abstractmethod
    def reset_accumulated_data(self):
        pass  
    @abstractmethod
    def get_config(self):
        pass

class FocalLoss(ILoss):
    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma or 0
        self._accumulated_loss_sum = None
        self._accumulated_mask_sum = None

    @property
    def accumulated_data(self):
        return self._accumulated_loss_sum,self._accumulated_mask_sum
        
    def reset_accumulated_data(self):
        self._accumulated_loss_sum = None
        self._accumulated_mask_sum = None
        
    def get_config(self):
        config = {}
        config['name'] = self.__class__.__name__
        config['gamma'] = self.gamma
        return config
        
    def forward(self, output, answer, mask, accumulate=False, **kwargs):
        """data shape is N,C,L"""
        """data shape is N,C,L"""
        if len(output.shape) != 3 or len(answer.shape) != 3:
            raise Exception("Wrong input shape",output.shape,answer.shape)

        if output.shape[:2] != answer.shape[:2]:
            raise Exception("Inconsist batch size or channel size",output.shape,answer.shape)
            
        if (output.shape[0],output.shape[2]) != mask.shape:
            raise Exception("Inconsist batch size or channel size",output.shape,mask.shape)

        _,C,L = output.shape
        mask = mask[:,:L].contiguous().view(-1).float()
        answer = answer[:,:,:L].transpose(1,2).reshape(-1,C).float()
        output = output.transpose(1,2).reshape(-1,C).float()
        loss =  (output+EPSILON).log()
        if self.gamma != 0:
            loss = (1-output)**self.gamma * loss
        loss = (-loss * answer).sum(1)
        loss = loss*mask
        loss_sum = loss.sum()
        mask_sum = mask.sum()

        if not accumulate:
            self._accumulated_loss_sum = loss_sum
            self._accumulated_mask_sum = mask_sum
        else:
            #Initialized if it is None
            self._accumulated_loss_sum = self._accumulated_loss_sum or 0
            self._accumulated_mask_sum = self._accumulated_mask_sum or 0
            self._accumulated_loss_sum = self._accumulated_loss_sum + loss_sum.detach()
            self._accumulated_mask_sum = self._accumulated_mask_sum + mask_sum.detach()
        
        loss = self._accumulated_loss_sum/(self._accumulated_mask_sum+EPSILON)
        return loss

class CCELoss(nn.Module):
    """Categorical cross entropy"""
    def __init__(self):
        super().__init__()
        self._loss = FocalLoss(0)

    @property
    def accumulated_data(self):
        return self._loss.accumulated_data
        
    def reset_accumulated_data(self):
        self._loss.reset_accumulated_data()
        
    def get_config(self):
        config = {}
        config['name'] = self.__class__.__name__
        return config
        
    def forward(self, output, answer, mask, **kwargs):
        """data shape is N,C,L"""
        loss =  self._loss(output,answer, mask, **kwargs)
        return loss

class SeqAnnLoss(nn.Module):
    def __init__(self,intron_coef=None,other_coef=None):
        super().__init__()
        self.intron_coef = intron_coef or 1
        self.other_coef = other_coef or 1
        self._accumulated_intron_loss_sum = None
        self._accumulated_other_loss_sum = None
        self._accumulated_mask_sum = None
        self._accumulated_transcript_mask_sum = None
        
    @property
    def accumulated_data(self):
        return (self._accumulated_intron_loss_sum,
                self._accumulated_other_loss_sum,
                self._accumulated_mask_sum,
                self._accumulated_transcript_mask_sum)
        
    def reset_accumulated_data(self):
        self._accumulated_intron_loss_sum = None
        self._accumulated_other_loss_sum = None
        self._accumulated_mask_sum = None
        self._accumulated_transcript_mask_sum = None
        
    def get_config(self):
        config = {}
        config['name'] = self.__class__.__name__
        config['intron_coef'] = self.intron_coef
        config['other_coef'] = self.other_coef
        return config
        
    def _calculate_mean(self,other_loss_sum,intron_loss_sum,mask_sum,transcript_mask_sum):
        other_loss = other_loss_sum/(mask_sum + EPSILON)
        intron_loss = intron_loss_sum/(transcript_mask_sum + EPSILON)
        loss = other_loss + intron_loss
        loss = loss / (self.intron_coef+self.other_coef)
        return loss
        
    def forward(self, output, answer, mask, accumulate=False,**kwargs):
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

        N,_,L = output.shape
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
        transcript_mask = mask*transcript_answer
        #Calculate loss
        other_loss = bce_loss(other_output,other_answer,return_mean=False)*self.other_coef
        intron_loss = bce_loss(intron_transcript_output,intron_answer,return_mean=False)*self.intron_coef*transcript_mask
        #Calculate sum
        other_loss_sum = sum_by_mask(other_loss,mask)
        intron_loss_sum = sum_by_mask(intron_loss,mask)
        mask_sum = mask.sum()
        transcript_mask_sum = transcript_mask.sum()
        if not accumulate:
            self._accumulated_intron_loss_sum = intron_loss_sum
            self._accumulated_other_loss_sum = other_loss_sum
            self._accumulated_mask_sum = mask_sum
            self._accumulated_transcript_mask_sum = transcript_mask_sum
        else:
            #Initialized if it is None
            self._accumulated_intron_loss_sum = self._accumulated_intron_loss_sum or 0
            self._accumulated_other_loss_sum = self._accumulated_other_loss_sum or 0
            self._accumulated_mask_sum = self._accumulated_mask_sum or 0
            self._accumulated_transcript_mask_sum = self._accumulated_transcript_mask_sum or 0
            ##Add value
            self._accumulated_intron_loss_sum = self._accumulated_intron_loss_sum + intron_loss_sum.detach()
            self._accumulated_other_loss_sum = self._accumulated_other_loss_sum + other_loss_sum.detach()
            self._accumulated_mask_sum = self._accumulated_mask_sum + mask_sum.detach()
            self._accumulated_transcript_mask_sum = self._accumulated_transcript_mask_sum + transcript_mask_sum.detach()
            
        loss = self._calculate_mean(self._accumulated_other_loss_sum,self._accumulated_intron_loss_sum,
                                    self._accumulated_mask_sum,self._accumulated_transcript_mask_sum)
        return loss

class LabelLoss(nn.Module):
    def __init__(self,loss):
        super().__init__()
        self.loss = loss
        self.predict_inference = create_basic_inference(3)
        self.answer_inference = create_basic_inference(3)
        
    @property
    def accumulated_data(self):
        return self.loss.accumulated_data
        
    def reset_accumulated_data(self):
        self.loss.reset_accumulated_data()
        
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
        loss = self.loss(label_predict,label_answer, mask,**kwargs)
        return loss
