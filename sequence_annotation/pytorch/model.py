from abc import ABCMeta
from torch import nn
from torch.nn.init import normal_,constant_
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch
from .noisy_activation import NoisyReLU
from .customize_layer import Conv1d, Concat,PaddedBatchNorm1d

class BasicModel(nn.Module,metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._out_channels = None
        self._distribution = {}
    @property
    def saved_distribution(self):
        return self._distribution

    def get_config(self):
        return {}

    @property
    def out_channels(self):
        return self._out_channels

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer,'reset_parameters'):
                layer.reset_parameters()

class BranchRNN(BasicModel):
    def __init__(self,in_channels,rnn_class,rnn_setting):
        super().__init__()
        self.rnn = rnn_class(in_channels,**rnn_setting)
        if hasattr(self.rnn,'out_channels'):
            out_size = self.rnn.out_channels
        else:
            out_size = self.rnn.hidden_size
        if self.rnn.bidirectional:
            out_size *= 2
        self.project_layer = Conv1d(out_size,out_channels=1,kernel_size=1)
        self._out_channels = out_size

    def forward(self,x):
        x,_ = self.rnn(x)
        projected_x = x
        projected_x,_ = pad_packed_sequence(projected_x, batch_first=True)
        projected_x = projected_x.transpose(1,2)
        projected_x = self.project_layer(projected_x)
        return projected_x,x

class HierachyRNN(BasicModel):
    def __init__(self,in_channels,rnns_type,rnn_setting,num_layers):
        super().__init__()
        self.rnns = []
        self.in_channels = in_channels
        for index in range(num_layers):
            rnn = BranchRNN(in_channels,rnns_type,rnn_setting)
            setattr(self,"rnn_{}".format(index),rnn)
            self.rnns.append(rnn)
            in_channels = rnn.out_channels
        self._out_channels = num_layers
        self.rnns_type = rnns_type
        self.rnn_setting = rnn_setting
        self.num_layers = num_layers
        self.norm = PaddedBatchNorm1d(num_layers)#nn.LayerNorm(self.out_channels)

    def forward(self,x,lengths):
        self._distribution["pre_rnn_result"] = x
        x = x.transpose(1,2)
        x = pack_padded_sequence(x,lengths, batch_first=True)
        projected_xs = []
        for rnn in self.rnns:
            projected_x,x = rnn(x)
            projected_xs.append(projected_x)
        result = torch.cat(projected_xs,dim=1)
        result = self.norm(result,lengths)
        self._distribution["rnn_result"] = result
        return result,lengths

    def get_config(self):
        return {'in_channels':self.in_channels,'rnns_type':str(self.rnns_type),
                'rnn_setting':self.rnn_setting,'num_layers':self.num_layers}

class CANBlock(BasicModel):
    def __init__(self,in_channels,kernel_size,out_channels,
                 norm_mode=None,norm_type=None,
                 activation_function=None):
        super().__init__()
        self.name = ""
        self.norm_type = norm_type
        if norm_mode in ["before_cnn","after_cnn","after_activation",None]:
            self.norm_mode = norm_mode
        else:
            raise Exception('norm_mode should be "before_cnn", "after_cnn", None, or "after_activation"')
        if activation_function is None:
            self.activation_function = NoisyReLU()
        else:
            self.activation_function = activation_function
        use_bias = self.norm_mode != "after_cnn"
        self.cnn = Conv1d(in_channels=in_channels,kernel_size=kernel_size,
                          out_channels=out_channels,bias=use_bias)
        if self.norm_mode is not None:
            in_channel = {"before_cnn":in_channels,"after_cnn":out_channels,"after_activation":out_channels}
            if self.norm_type is None:
                self.norm = PaddedBatchNorm1d(in_channel[self.norm_mode])
            else:
                self.norm = self.norm_type(in_channel[self.norm_mode])
        self._out_channels = out_channels
        self._distribution = {}
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.children():
            if isinstance(layer,Conv1d):
                constant_(layer.bias,0)
                normal_(layer.weight,0,0.01)

    def _normalized(self,x,lengths):
        x = self.norm(x,lengths)
        self._distribution['norm_{}'.format(self.name)] = x
        return x

    def forward(self, x, lengths):
        #X shape : N,C,L
        x_ = x
        if self.norm_mode=='before_cnn':
            x_ = self._normalized(x_,lengths)
        x_,lengths = self.cnn(x_,lengths)
        self._distribution['cnn_x_{}'.format(self.name)] = x_
        if self.norm_mode=='after_cnn':
            x_ = self._normalized(x_,lengths)
        x_ = self.activation_function(x_)
        self._distribution['post_act_x_{}'.format(self.name)] = x_
        if self.norm_mode=='after_activation':
            x_ = self._normalized(x_,lengths)
        return x_,lengths

class ConcatCNN(BasicModel):
    def __init__(self,in_channels,num_layers,cnns_setting):
        super().__init__()
        self.num_layers = num_layers
        in_num = in_channels
        self.cnns = []
        for index in range(self.num_layers):
            setting = {}
            for key,value in cnns_setting.items():
                if isinstance(value,list):
                    setting[key] = value[index]
                else:
                    setting[key] = value
            cnn = CANBlock(in_num,**setting)
            cnn.name=str(index)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            in_num += cnn.out_channels
        self._out_channels = in_num
        self.concat = Concat(dim=1)
        self.reset_parameters()

    def forward(self, x, lengths):
        #X shape : N,C,L
        for index in range(self.num_layers):
            pre_x = x
            cnn = self.cnns[index]
            x,lengths = cnn(x,lengths)
            self._distribution.update(cnn.saved_distribution)
            x,lengths = self.concat([pre_x,x],lengths)
        self._distribution['cnn_result'] = x
        return x,lengths

class ProjectLayer(BasicModel):
    def __init__(self,in_channels,out_channels,kernel_size=None):
        super().__init__()
        self.cnn = Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size or 1)
        self._out_channels = out_channels

    def forward(self,x,lengths):
        x = self.cnn(x)
        return x,lengths

    def get_config(self):
        config = {'out_channels':self.cnn.out_channels,
                  'kernel_size':self.cnn.kernel_size}
        return config

    def reset_parameters(self):
        constant_(self.cnn.bias,0)
        normal_(self.cnn.weight,0,0.1)


class FeatureBlock(BasicModel):
    def __init__(self,input_size,num_layers,cnns_setting,reduce_cnn_ratio=None,reduced_cnn_number=None):
        super().__init__()
        self.input_size=input_size
        self.cnns_setting = cnns_setting
        self.num_layers = num_layers
        if self.num_layers <= 0:
            raise Exception("CNN layer size sould be positive")
        self.reduced_cnn_number = reduced_cnn_number
        self.reduce_cnn_ratio = reduce_cnn_ratio or 1
        if self.reduce_cnn_ratio <= 0:
            raise Exception("Reduce_cnn_ratio should be larger than zero")
        self._reduce_cnn = self.reduce_cnn_ratio < 1 or self.reduced_cnn_number is not None
        self._build_layers()
        self.reset_parameters()
        self.save_distribution = True

    def get_config(self):
        config = {}
        config['input_size'] = self.input_size
        setting = self.cnns_setting
        for type_ in ['norm_type','activation_function']:
            if type_ in setting.keys():
                setting[type_] = str(setting[type_])
        config['cnns_setting'] = setting
        config['reduce_cnn_ratio'] = self.reduce_cnn_ratio
        config['reduced_cnn_number'] = self.reduced_cnn_number
        config['save_distribution'] = self.save_distribution
        return config

    def _build_layers(self):
        input_size=self.input_size
        self.cnns = ConcatCNN(input_size,self.num_layers,self.cnns_setting)
        self.cnn_ln = nn.LayerNorm(self.cnns.out_channels)
        if self._reduce_cnn:
            if self.reduce_cnn_ratio < 1:
                hidden_size = int(self.cnns.hidden_size*self.reduce_cnn_ratio)
            else:
                hidden_size = self.reduced_cnn_number
            self.cnn_merge = Conv1d(in_channels=self.cnns.hidden_size,
                                    kernel_size=1,out_channels=hidden_size)
            input_size = hidden_size
        else:
            input_size = self.cnns.out_channels
        self._out_channels = input_size

    def forward(self, x, lengths):
        #X shape : N,C,L
        x,lengths = self.cnns(x,lengths)
        cnn_distribution = self.cnns.saved_distribution
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
        self.rnns_type = rnns_type
        self._build_layers()
        self.reset_parameters()
        self.save_distribution = True

    def get_config(self):
        config = {}
        config['input_size'] = self.input_size
        rnns_setting = self.rnns_setting
        if 'rnns_type' in rnns_setting.keys():
            rnns_setting['rnns_type'] = str(rnns_setting['rnns_type'])
        config['rnns_setting'] = rnns_setting
        config['rnns_type'] = str(self.rnns_type)
        config['save_distribution'] = self.save_distribution
        return config

    def _build_layers(self):
        input_size=self.input_size
        self.rnns = self.rnns_type(input_size=input_size,**self.rnns_setting)
        if hasattr(self.rnns,'out_channels'):
            input_size = self.rnns.out_channels
        else:
            input_size=self.rnns.hidden_size
            if self.rnns.bidirectional:
                input_size *= 2
        self._out_channels = input_size

    def forward(self, x, lengths):
        #X shape : N,C,L
        self._distribution["pre_rnn_result"] = x
        rnn_distribution = {}
        x = x.transpose(1,2)
        x = pack_padded_sequence(x,lengths, batch_first=True)
        x,_ = self.rnns(x)
        x,_ = pad_packed_sequence(x, batch_first=True)
        x = x.transpose(1,2)
        self._distribution["pre_last_result"] = x
        if self.save_distribution:
            self._distribution.update(rnn_distribution)
        return x,lengths

class SeqAnnModel(BasicModel):
    def __init__(self,feature_block,relation_block,project_layer,use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
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
        config['project_layer'] = self.project_layer.get_config()
        return config

    def forward(self, x, lengths):
        #X shape : N,C,L
        x,lengths = self.feature_block(x, lengths)
        x,lengths = self.relation_block(x, lengths)
        self._distribution["pre_last"] = x
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

class SAGAN(nn.Module):
    """Sequence annotation GAN model"""
    def __init__(self,gan,discrim):
        super().__init__()
        self.gan = gan
        self.discrim = discrim

def seq_ann_inference(outputs):
    """
        Data shape is N,C,L
        Output channel order: Transcription potential, Intron potential
    """
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

def seq_ann_alt_inference(outputs):
    """
        Data shape is N,C,L
        Inputs channel order: Transcription potential, Intron|Transcription potential,
                              Alternative intron|Intron potential
    """
    if outputs.shape[1] != 3:
        raise Exception("Channel size should be three")
    transcript_potential = (outputs[:,0,:].unsqueeze(1)>=0.5).float().cuda()
    intron_trans_potential = (outputs[:,1,:].unsqueeze(1)>=0.5).float().cuda()
    alt_intron_potential = (outputs[:,2,:].unsqueeze(1)>=0.5).float().cuda()

    other = 1-transcript_potential
    exon = transcript_potential*(1-intron_trans_potential)
    intron = transcript_potential*intron_trans_potential
    nonalt_intron = intron * (1-alt_intron_potential)
    alt_intron = intron * alt_intron_potential

    result = torch.cat([alt_intron,exon,nonalt_intron,other],dim=1)
    #print(transcript_potential[0,0,0],intron_trans_potential[0,0,0],alt_intron_potential[0,0,0],result[0,:,0])
    return result
