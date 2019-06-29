from abc import abstractproperty,ABCMeta,abstractmethod
from .CRF import BatchCRF
from .customize_layer import Conv1d, ConcatCNN, PWM, ConcatRNN
from torch import nn
from torch.nn.init import ones_,zeros_,uniform_,normal_,constant_,eye_
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch

class BasicModel(nn.Module,metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._out_channels = None
        
    @property
    def saved_distribution(self):
        return self._distribution

    @abstractmethod
    def get_config(self):
        pass

    @property
    def out_channels(self):
        return self._out_channels

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer,'reset_parameters'):
                layer.reset_parameters()
    
def init_GRU(gru):
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

def seq_ann_inference(outputs):
    #N,C,L
    if outputs.shape[1]!=2:
        raise Exception("Channel size should be two")
    transcript_potential = outputs[:,0,:].unsqueeze(1)
    intron_potential = outputs[:,1,:].unsqueeze(1)
    other = 1-transcript_potential
    transcript_mask = (transcript_potential>=0.5).float()
    intron = transcript_mask*intron_potential
    exon = transcript_mask*(1-intron_potential)
    result = torch.cat([exon,intron,other],dim=1)
    return result
            
class FeatureBlock(BasicModel):
    def __init__(self,input_size,cnns_setting,reduce_cnn_ratio=None,reduced_cnn_number=None):
        super().__init__()
        self.input_size=input_size
        self.cnns_setting = cnns_setting
        if self.cnns_setting['num_layers'] <= 0:
            raise Exception("CNN layer size sould be positive")
        self.reduced_cnn_number = reduced_cnn_number
        self.reduce_cnn_ratio = reduce_cnn_ratio or 1
        if self.reduce_cnn_ratio <= 0:
            raise Exception("Reduce_cnn_ratio should be larger than zero")
        self._distribution = {}
        self._reduce_cnn = self.reduce_cnn_ratio < 1 or self.reduced_cnn_number is not None
        self._build_layers()
        self.reset_parameters()
        self.save_distribution = True

    def get_config(self):
        config = {}
        config['input_size'] = self.input_size
        config['cnns_setting'] = self.cnns_setting
        config['reduce_cnn_ratio'] = self.reduce_cnn_ratio
        config['reduced_cnn_number'] = self.reduced_cnn_number
        config['save_distribution'] = self.save_distribution
        return config
        
    def _build_layers(self):
        input_size=self.input_size
        self.cnns = ConcatCNN(input_size,**self.cnns_setting)
        self.cnn_ln = nn.LayerNorm(self.cnns.hidden_size)
        if self._reduce_cnn:
            if self.reduce_cnn_ratio < 1:
                hidden_size = int(self.cnns.hidden_size*self.reduce_cnn_ratio)
            else:
                hidden_size = self.reduced_cnn_number
            self.cnn_merge = Conv1d(in_channels=self.cnns.hidden_size,
                                    kernel_size=1,out_channels=hidden_size)
            input_size = hidden_size    
        else:
            input_size = self.cnns.hidden_size
        self._out_channels = input_size
        
    def forward(self, x, lengths):
        #X shape : N,C,L
        x,lengths,cnn_distribution = self.cnns(x,lengths)
        x = self.cnn_ln(x.transpose(1,2)).transpose(1,2)
        if self._reduce_cnn:
            x,lengths = self.cnn_merge(x,lengths)
        if self.save_distribution:
            self._distribution.update(cnn_distribution)
        return x,lengths

class RelationBlock(BasicModel):
    def __init__(self,input_size,rnns_setting,rnns_type=None):
        super().__init__()
        self.input_size=input_size
        self.rnns_setting = rnns_setting
        self.rnn_num = self.rnns_setting['num_layers']
        if self.rnns_setting['num_layers'] <= 0:
            raise Exception("RNN layer size sould be positive")
        self.rnns_type = rnns_type or 'LSTM'
        self._distribution = {}
        self._build_layers()
        self.reset_parameters()
        self.save_distribution = True

    def get_config(self):
        config = {}
        config['input_size'] = self.input_size
        config['rnns_setting'] = self.rnns_setting
        config['rnns_type'] = self.rnns_type
        config['save_distribution'] = self.save_distribution
        return config
        
    def _build_layers(self):
        input_size=self.input_size
        if self.rnns_type == 'LSTM':
            rnn_class = nn.LSTM
        elif self.rnns_type == 'GRU':
            rnn_class = nn.GRU
        elif self.rnns_type == 'customized':
            rnn_class = ConcatRNN
        self.rnns = rnn_class(input_size=input_size,**self.rnns_setting)
        input_size=self.rnns.hidden_size
        if self.rnns_type in ['LSTM','GRU']:
            if self.rnns.bidirectional:
                input_size *= 2
        self._out_channels = input_size

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.children():
            if isinstance(layer,nn.GRU):
                init_GRU(layer)
        
    def forward(self, x, lengths):
        #X shape : N,C,L
        distribution_output = {}
        self._distribution["pre_rnn_result"] = x
        if self.rnns_type != 'customized':
            rnn_distribution = {}
            x = x.transpose(1,2)
            x = pack_padded_sequence(x,lengths, batch_first=True)
            x,_ = self.rnns(x)
            x,_ = pad_packed_sequence(x, batch_first=True)
            x = x.transpose(1,2)
        else:
            x,lengths,rnn_distribution =self.rnns(x,lengths)
        self._distribution["pre_last_result"] = x
        if self.save_distribution:
            self._distribution.update(rnn_distribution)
        return x,lengths

class SeqAnnModel(BasicModel):
    def __init__(self,feature_block,relation_block,project_layer,use_sigmoid=None):
        super().__init__()
        self.use_sigmoid = use_sigmoid or False
        self._distribution = {}
        self.feature_block = feature_block
        self.relation_block = relation_block
        self.project_layer = project_layer
        self._out_channels = self.project_layer.out_channels
        self.reset_parameters()

    def get_config(self):
        config = {}
        config['use_sigmoid'] = self.use_sigmoid
        config['feature_block'] = self.feature_block.get_config()
        config['relation_block'] = self.relation_block.get_config()
        config['project_layer'] = {'out_channels':self.project_layer.out_channels,
                                   'kernel_size':self.project_layer.kernel_size}
        return config

    def forward(self, x, lengths):
        #X shape : N,C,L
        x,lengths = self.feature_block(x, lengths)
        x,lengths = self.relation_block(x, lengths)
        x,lengths = self.project_layer(x,lengths)   
        self._distribution.update(self.feature_block.saved_distribution)
        self._distribution.update(self.relation_block.saved_distribution)
        self._distribution["last"] = x
        if not self.use_sigmoid:
            x = nn.LogSoftmax(dim=1)(x).exp()
            self._distribution["softmax"] = x
        else:
            x = torch.sigmoid(x)
            self._distribution["sigmoid"] = x
        return x

    def reset_parameters(self):
        super().reset_parameters()
        constant_(self.project_layer.bias,0)
        normal_(self.project_layer.weight)

class SAGAN(nn.Module):
    """Sequence annotation GAN model"""
    def __init__(self,gan,discrim):
        super().__init__()
        self.gan = gan
        self.discrim = discrim
