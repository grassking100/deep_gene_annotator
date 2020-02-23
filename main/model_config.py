import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import write_json
from sequence_annotation.process.cnn import PADDING_HANDLE

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("--config_path",help="Path to save config",required=True)
    parser.add_argument("--cnn_act",default='NoisyReLU')
    parser.add_argument("--cnn_out",type=int,default=16)
    parser.add_argument("--cnn_kernel",type=int,default=16)
    parser.add_argument("--cnn_num",type=int,default=4)
    parser.add_argument("--stack_cnn_class",type=str,default='ConcatCNN')
    parser.add_argument("--rnn_size",type=int,default=16)
    parser.add_argument("--rnn_num",type=int,default=4)
    parser.add_argument("--rnn_type",type=str,default='GRU')
    parser.add_argument("--customized_cnn",default='xavier_uniform_cnn_init')
    parser.add_argument("--customized_rnn_init",default='in_xav_bias_zero_gru_init')
    parser.add_argument("--customized_rnn_cnn",default='xavier_uniform_cnn_init')
    parser.add_argument("--padding_handle",default='valid',
                        help="Handle padding issue, valid "
                        "options are {}".format(', '.join(PADDING_HANDLE)))
    parser.add_argument("--padding_value",type=float,default=0)
    parser.add_argument("--bottleneck_factor",type=float)
    parser.add_argument("--norm_mode",default='after_activation')
    parser.add_argument("--norm_type",type=str)
    parser.add_argument("--out_channels",type=int,default=3)
    parser.add_argument("--feature_dropout",type=float,default=0)
    parser.add_argument("--relation_dropout",type=float,default=0)
    parser.add_argument("--use_common_atten",action='store_true')
    parser.add_argument("--not_use_first_atten",action='store_true')
    parser.add_argument("--not_use_second_atten",action='store_true')
    parser.add_argument("--atten_hidden_size",type=int,default=None)
    parser.add_argument("--atten_num_layers",type=int,default=None)
    parser.add_argument("--not_norm_input",action='store_true')
    parser.add_argument("--norm_momentum",type=float,default=None)
    parser.add_argument("--norm_affine",action='store_true')
    parser.add_argument("--hier_option")
    parser.add_argument("--output_act")
    
    args = parser.parse_args()

    config = {}
    feature_block_config = {}
    relation_block_config = {}
    feature_block_config['dropout'] = args.feature_dropout
    feature_block_config['stack_cnn_class'] = args.stack_cnn_class
    feature_block_config['num_layers'] = args.cnn_num
    feature_block_config['norm_mode'] = args.norm_mode
    feature_block_config['norm_type'] = args.norm_type
    feature_block_config['out_channels'] = args.cnn_out
    feature_block_config['kernel_size'] = args.cnn_kernel
    feature_block_config['activation_function'] = args.cnn_act
    feature_block_config['padding_handle'] = args.padding_handle
    feature_block_config['padding_value'] = args.padding_value
    feature_block_config['customized_init'] = args.customized_cnn
    feature_block_config['norm_input'] = not args.not_norm_input
    feature_block_config['norm_momentum'] = args.norm_momentum
    feature_block_config['norm_affine'] = args.norm_affine
    feature_block_config['bottleneck_factor'] = args.bottleneck_factor
    
    relation_block_config['rnn_type'] = args.rnn_type
    relation_block_config['num_layers'] = args.rnn_num
    relation_block_config['hidden_size'] = args.rnn_size
    relation_block_config['customized_rnn_init'] = args.customized_rnn_init
    relation_block_config['customized_cnn_init'] = args.customized_rnn_cnn
    relation_block_config['dropout'] = args.relation_dropout
    relation_block_config['use_common_atten'] = args.use_common_atten
    relation_block_config['use_first_atten'] = not args.not_use_first_atten
    relation_block_config['use_second_atten'] = not args.not_use_second_atten
    relation_block_config['atten_hidden_size'] = args.atten_hidden_size
    relation_block_config['atten_num_layers'] = args.atten_num_layers
    relation_block_config['hier_option'] = args.hier_option
    relation_block_config['output_act'] = args.output_act
    relation_block_config['out_channels'] = args.out_channels
    
    config['feature_block_config'] = feature_block_config
    config['relation_block_config'] = relation_block_config
    
    write_json(config,args.config_path)
