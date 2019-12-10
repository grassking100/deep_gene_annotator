import sys
sys.path.append("/home/sequence_annotation")
from sequence_annotation.process.model import SeqAnnBuilder

def _add_cnn(trial,builder,cnn_num,cnn_out,radius):
    if cnn_num > 0 :
        builder.feature_block_config['num_layers'] = cnn_num*(2**trial.suggest_int('cnn_num_coef',0,3))
        builder.feature_block_config['out_channels'] = cnn_out*(2**trial.suggest_int('cnn_out_coef',0,3))
        builder.feature_block_config['kernel_size'] = 2*(radius*(4**trial.suggest_int('radius_coef',0,4)))+1

def _add_rnn(trial,builder,rnn_type,rnn_hidden,rnn_hidden_coef_max):
    if rnn_hidden_coef_max >= 0:
        builder.relation_block_config['hidden_size'] = rnn_hidden*(2**trial.suggest_int('rnn_hidden_coef',0,rnn_hidden_coef_max))


class Builder:
    def __init__(self):
        self.cnn_num = 2
        self.cnn_out = 4
        self.padding_handle = 'same'
        self.rnn_hidden = 16
        self.radius = 1
        self.rnn_type = 'ProjectedGRU'
        self.rnn_hidden_coef_max = 3
        
    def build(self,out_channels,last_act,
              cnn_num=None,cnn_out=None,radius=None,
              padding_handle=None,
              rnn_hidden=None,rnn_type=None,
              norm_input=True,norm_affine=True,
              rnn_hidden_coef_max=None,
              use_first_atten=True,use_second_atten=True,from_trial_params=False):
        if cnn_num == 0:
            cnn_num = 0
        else:
            cnn_num = cnn_num or self.cnn_num
        cnn_out = cnn_out or self.cnn_out
        radius = radius or self.radius
        padding_handle = padding_handle or self.padding_handle
        rnn_hidden = rnn_hidden or self.rnn_hidden
        rnn_type = rnn_type or self.rnn_type
        
        if rnn_hidden_coef_max == 0:
            rnn_hidden_coef_max = 0
        else:
            rnn_hidden_coef_max = rnn_hidden_coef_max or self.rnn_hidden_coef_max

        builder = SeqAnnBuilder()
        builder.feature_block_config['num_layers'] = 0
        builder.feature_block_config['norm_input'] = norm_input
        builder.feature_block_config['norm_affine'] = norm_affine
        builder.relation_block_config['use_first_atten'] = use_first_atten
        builder.relation_block_config['use_second_atten'] = use_second_atten
        builder.out_channels = out_channels
        builder.last_act = last_act
        
        builder.feature_block_config['customized_init'] = 'xavier_uniform_cnn_init'
        builder.feature_block_config['padding_handle'] = padding_handle
        builder.relation_block_config['rnn_type'] = rnn_type
        builder.relation_block_config['num_layers'] = 2
        builder.relation_block_config['hidden_size'] = rnn_hidden
        builder.relation_block_config['customized_gru_init'] = 'in_xav_bias_zero_gru_init'
        builder.relation_block_config['customized_cnn_init'] = 'xavier_uniform_cnn_init'
        
        def generator(trial):
            if not from_trial_params:
                _add_cnn(trial,builder,cnn_num,cnn_out,radius)
                _add_rnn(trial,builder,rnn_type,rnn_hidden,rnn_hidden_coef_max)
            else:
                builder.feature_block_config['num_layers'] = cnn_num*(2**trial.params['cnn_num_coef'])
                builder.feature_block_config['out_channels'] = cnn_num*(2**trial.params['cnn_out_coef'])
                builder.feature_block_config['kernel_size'] = 2*(radius*(4**trial.params['radius_coef']))+1
                builder.relation_block_config['hidden_size'] = rnn_hidden*(2**trial.params['rnn_hidden_coef'])
            return builder.config
        return generator
        