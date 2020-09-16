import abc
import torch
import numpy as np

def ann2one_hot(x):
    N,C,L = x.shape
    x = x.transpose(0,2,1).reshape(-1,C)
    y = np.zeros(x.shape)
    y[np.arange(N*L),x.argmax(1)]=1
    y = y.reshape(N,L,C).transpose(0,2,1)
    return y

class AbstractInference(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self):
        pass
    
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        return config

class BasicInference(AbstractInference):
    """Convert vectors into one-hot encoding"""
    def __init__(self,first_n_channel):
        self._first_n_channel = first_n_channel
        
    def get_config(self):
        config = super().get_config()
        config['first_n_channel'] = self._first_n_channel
        return config
        
    def __call__(self, ann):
        """
            Input shape is N,C,L (where C>=2)
            Output shape is N,FisrtN,L
        """
        C = ann.shape[1]
        if self._first_n_channel > C:
            raise Exception("Got {} channel, but except {}".format(C,self._first_n_channel))
        ann = ann[:, :self._first_n_channel, :]
        onehot = torch.FloatTensor(ann2one_hot(ann.cpu().numpy()))
        return onehot

class SeqAnnInference(AbstractInference):
    """Convert vectors into one-hot encoding of gene annotation"""
    def __init__(self,transcript_threshold=None,intron_threshold=None):
        self._transcript_threshold = transcript_threshold or 0.5
        self._intron_threshold = intron_threshold or 0.5
        
    def get_config(self):
        config = super().get_config()
        config['transcript_threshold'] = self._transcript_threshold
        config['intron_threshold'] = self._intron_threshold
        return config
        
    def __call__(self,ann):
        """
            Input shape is N,C,L (where C>=2)
            Input channel order: Transcription potential, Intron potential
            Output channel order: Exon, Intron , Other
        """
        transcript_potential = ann[:, 0, :].unsqueeze(1)
        intron_potential = ann[:, 1, :].unsqueeze(1)
        transcript_mask = (transcript_potential >= self._transcript_threshold).float()
        intron_mask = (intron_potential >= self._intron_threshold).float()
        exon = transcript_mask * (1 - intron_mask)
        intron = transcript_mask * intron_mask
        other = 1 - transcript_mask
        result = torch.cat([exon, intron, other], dim=1)
        return result

INFERENCES_TYPES = {'BasicInference':BasicInference,'SeqAnnInference':SeqAnnInference}
    