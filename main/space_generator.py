import sys
sys.path.append("/home/sequence_annotation")
from sequence_annotation.process.model import SeqAnnBuilder

def _add_cnn(trial,builder,cnn_num,cnn_out,padding_handle):
    builder.feature_block_config['num_layers'] = cnn_num*(2**trial.suggest_int('cnn_num_coef',0,3))
    builder.feature_block_config['out_channels'] = cnn_out*(2**trial.suggest_int('cnn_out_coef',0,3))
    builder.feature_block_config['padding_handle'] = padding_handle
    builder.feature_block_config['kernel_size'] = 3
    builder.feature_block_config['customized_init'] = 'xavier_uniform_cnn_init'

def _add_rnn(trial,builder,rnn_type,rnn_hidden):
    builder.relation_block_config['rnn_type'] = rnn_type
    builder.relation_block_config['num_layers'] = 2
    builder.relation_block_config['hidden_size'] = rnn_hidden*(2**trial.suggest_int('rnn_hidden_coef',0,3))
    builder.relation_block_config['customized_gru_init'] = 'in_xav_bias_zero_gru_init'
    builder.relation_block_config['customized_cnn_init'] = 'xavier_uniform_cnn_init'

class Builder:
    def __init__(self):
        self.cnn_num = 2
        self.cnn_out = 4
        self.padding_handle = 'same'
        self.rnn_hidden = 16
        self.rnn_type = 'ProjectedGRU'
        
    def build(self,out_channels,last_act,
              cnn_num=None,cnn_out=None,padding_handle=None,
              rnn_hidden=None,rnn_type=None,
              norm_input=True,norm_affine=True):
        if cnn_num == 0:
            cnn_num = 0
        else:
            cnn_num = self.cnn_num or cnn_num
        cnn_out = cnn_out or self.cnn_out
        padding_handle = padding_handle or self.padding_handle
        rnn_hidden = rnn_hidden or self.rnn_hidden
        rnn_type = rnn_type or self.rnn_type
        builder = SeqAnnBuilder()
        builder.feature_block_config['num_layers'] = 0
        builder.feature_block_config['norm_input'] = norm_input
        builder.feature_block_config['norm_affine'] = norm_affine
        builder.out_channels = out_channels
        builder.last_act = last_act
        def generator(trial):
            if cnn_num > 0 :
                _add_cnn(trial,builder,cnn_num,cnn_out,padding_handle)
            _add_rnn(trial,builder,rnn_type,rnn_hidden)
            return builder.config
        return generator
        