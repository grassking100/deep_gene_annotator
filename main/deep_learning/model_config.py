import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/..")
from sequence_annotation.utils.utils import write_json,get_time_str
from sequence_annotation.process.cnn import PADDING_HANDLE

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("--config_path",help="Path to save config",required=True)
    parser.add_argument("--not_norm_input",action='store_true')
    #CNN config
    parser.add_argument("--customized_cnn_init",default='xavier_uniform_cnn_init')
    parser.add_argument("--cnn_act",default='NoisyReLU')
    parser.add_argument("--cnn_out",type=int,default=16)
    parser.add_argument("--cnn_kernel",type=int,default=16)
    parser.add_argument("--cnn_num",type=int,default=4)
    parser.add_argument("--stack_cnn_class",type=str,default='ConcatCNN')
    parser.add_argument("--padding_handle",default='valid',help="Handle padding issue, valid "\
                        "options are {}".format(', '.join(PADDING_HANDLE)))
    parser.add_argument("--bottleneck_factor",type=float)
    parser.add_argument("--feature_dropout",type=float,default=0)
    parser.add_argument("--norm_mode",default='after_activation')
    #Relation config
    parser.add_argument("--rnn_size",type=int,default=16)
    parser.add_argument("--rnn_num",type=int,default=4)
    parser.add_argument("--rnn_type",type=str,default='GRU')
    parser.add_argument("--customized_rnn_init",default='in_xav_bias_zero_gru_init')
    parser.add_argument("--customized_relation_cnn_init",default='xavier_uniform_cnn_init')
    parser.add_argument("--relation_dropout",type=float,default=0)
    parser.add_argument("--out_channels",type=int,default=3)
    parser.add_argument("--hier_option")
    parser.add_argument("--output_act")
    #Norm config
    parser.add_argument("--norm_class",type=str)
    parser.add_argument("--norm_momentum",type=float,default=None)
    parser.add_argument("--norm_affine",action='store_true')
    #Filter config
    parser.add_argument("--use_common_filter",action='store_true')
    parser.add_argument("--use_first_filter",action='store_true')
    parser.add_argument("--use_second_filter",action='store_true')
    parser.add_argument("--filter_hidden",type=int,default=None)
    parser.add_argument("--filter_num",type=int,default=None)

    args = parser.parse_args()

    config = {'use_input_norm':not args.not_norm_input}
    feature_block_config = {}
    relation_block_config = {}
    norm_config = {}
    
    #Feature config
    feature_block_config['customized_init'] = args.customized_cnn_init
    feature_block_config['stack_cnn_class'] = args.stack_cnn_class
    feature_block_config['num_layers'] = args.cnn_num
    feature_block_config['out_channels'] = args.cnn_out
    feature_block_config['kernel_size'] = args.cnn_kernel
    feature_block_config['activation_function'] = args.cnn_act
    feature_block_config['padding_handle'] = args.padding_handle
    feature_block_config['dropout'] = args.feature_dropout
    feature_block_config['bottleneck_factor'] = args.bottleneck_factor
    feature_block_config['norm_mode'] = args.norm_mode
    
    #Relation config
    relation_block_config['rnn_type'] = args.rnn_type
    relation_block_config['num_layers'] = args.rnn_num
    relation_block_config['hidden_size'] = args.rnn_size
    relation_block_config['customized_rnn_init'] = args.customized_rnn_init
    relation_block_config['customized_cnn_init'] = args.customized_relation_cnn_init
    relation_block_config['dropout'] = args.relation_dropout
    relation_block_config['hier_option'] = args.hier_option
    relation_block_config['output_act'] = args.output_act
    relation_block_config['out_channels'] = args.out_channels
    
    #Norm config
    norm_config['norm_class'] = args.norm_class
    norm_config['momentum'] = args.norm_momentum
    norm_config['affine'] = args.norm_affine

    #Filter config
    relation_block_config['use_common_filter'] = args.use_common_filter
    relation_block_config['use_first_filter'] = args.use_first_filter
    relation_block_config['use_second_filter'] = args.use_second_filter
    relation_block_config['filter_hidden'] = args.filter_hidden
    relation_block_config['filter_num'] = args.filter_num

    config['feature_block_config'] = feature_block_config
    config['relation_block_config'] = relation_block_config
    config['norm_config'] = norm_config
    config['generated_time'] = get_time_str()
    
    write_json(config,args.config_path)
