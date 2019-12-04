import os
import sys
from argparse import ArgumentParser
import json
sys.path.append("/home/sequence_annotation")
from sequence_annotation.process.model import SeqAnnBuilder
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
    parser.add_argument("--train_init_value",action="store_true")
    parser.add_argument("--customized_cnn")
    parser.add_argument("--customized_gru_init")
    parser.add_argument("--customized_rnn_cnn")
    parser.add_argument("--padding_handle",default='valid',
                        help='Handle padding issue, valid options are {}'.format(', '.join(PADDING_HANDLE)))
    parser.add_argument("--padding_value",type=float,default=0)
    parser.add_argument("--bottleneck_factor",type=float)
    parser.add_argument("--norm_mode",default='after_activation')
    parser.add_argument("--norm_type",type=str)
    parser.add_argument("--out_channels",type=int,default=3)
    parser.add_argument("--rnn_dropout",type=float,default=0)    
    parser.add_argument("--use_common_atten",action='store_true')
    parser.add_argument("--not_use_first_atten",action='store_true')
    parser.add_argument("--not_use_second_atten",action='store_true')
    parser.add_argument("--atten_hidden_size",type=int,default=None)
    parser.add_argument("--atten_num_layers",type=int,default=None)
    parser.add_argument("--norm_input",action='store_true')
    parser.add_argument("--norm_momentum",type=float,default=None)
    parser.add_argument("--not_norm_affine",action='store_true')
    parser.add_argument("--hier_option")
    
    parser.add_argument("--last_act",type=str)
    
    args = parser.parse_args()

    builder = SeqAnnBuilder()
    builder.feature_block_config['stack_cnn_class'] = args.stack_cnn_class
    builder.feature_block_config['num_layers'] = args.cnn_num
    builder.feature_block_config['norm_mode'] = args.norm_mode
    builder.feature_block_config['norm_type'] = args.norm_type
    builder.feature_block_config['out_channels'] = args.cnn_out
    builder.feature_block_config['kernel_size'] = args.cnn_kernel
    builder.feature_block_config['activation_function'] = args.cnn_act
    builder.feature_block_config['padding_handle'] = args.padding_handle
    builder.feature_block_config['padding_value'] = args.padding_value
    builder.feature_block_config['customized_init'] = args.customized_cnn
    builder.feature_block_config['norm_input'] = args.norm_input
    builder.feature_block_config['norm_momentum'] = args.norm_momentum
    builder.feature_block_config['norm_affine'] = not args.not_norm_affine
    builder.feature_block_config['bottleneck_factor'] = args.bottleneck_factor
    
    builder.relation_block_config['rnn_type'] = args.rnn_type
    builder.relation_block_config['num_layers'] = args.rnn_num
    builder.relation_block_config['hidden_size'] = args.rnn_size
    builder.relation_block_config['train_init_value'] = args.train_init_value
    builder.relation_block_config['customized_gru_init'] = args.customized_gru_init
    builder.relation_block_config['customized_cnn_init'] = args.customized_rnn_cnn
    builder.relation_block_config['dropout'] = args.rnn_dropout
    builder.relation_block_config['use_common_atten'] = args.use_common_atten
    builder.relation_block_config['use_first_atten'] = not args.not_use_first_atten
    builder.relation_block_config['use_second_atten'] = not args.not_use_second_atten
    builder.relation_block_config['atten_hidden_size'] = args.atten_hidden_size
    builder.relation_block_config['atten_num_layers'] = args.atten_num_layers
    builder.relation_block_config['hier_option'] = args.hier_option

    builder.out_channels = args.out_channels
    builder.last_act = args.last_act
    
    with open(args.config_path,"w") as fp:
        json.dump(builder.config, fp, indent=4)
