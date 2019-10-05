import os
import sys
from argparse import ArgumentParser
import json
sys.path.append("/home/sequence_annotation")
from sequence_annotation.process.model import SeqAnnBuilder
from sequence_annotation.process.customized_layer import PADDING_HANDLE

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
    parser.add_argument("--use_discrim",action="store_true")
    parser.add_argument("--disrim_rnn_size",type=int,default=16)
    parser.add_argument("--disrim_rnn_num",type=int,default=1)
    parser.add_argument("--customized_gru_init_mode")
    parser.add_argument("--padding_handle",help='Handle padding issue, valid options are {}'.format(', '.join(PADDING_HANDLE)),
                        default='valid')
    parser.add_argument("--padding_value",type=float,default=0)
    parser.add_argument("--bottleneck_factor",type=float)
    parser.add_argument("--compression_factor",type=float)
    parser.add_argument("--feature_dropout",type=float)
    parser.add_argument("--norm_mode",default='after_activation')
    parser.add_argument("--site_ann_method")
    parser.add_argument("--predict_site_by")
    parser.add_argument("--out_channels",type=int,default=3)
    parser.add_argument("--use_sigmoid",action="store_true")
    parser.add_argument("--project_kernel_size",type=int,default=1)
    parser.add_argument("--rnn_dropout",type=float,default=0)
    
    
    args = parser.parse_args()

    builder = SeqAnnBuilder()
    builder.feature_block_config['stack_cnn_class'] = args.stack_cnn_class
    builder.feature_block_config['num_layers'] = args.cnn_num
    builder.feature_block_config['norm_mode'] = args.norm_mode
    builder.feature_block_config['cnn_setting']['out_channels'] = args.cnn_out
    builder.feature_block_config['cnn_setting']['kernel_size'] = args.cnn_kernel
    builder.feature_block_config['cnn_setting']['activation_function'] = args.cnn_act
    builder.feature_block_config['cnn_setting']['padding_handle'] = args.padding_handle
    builder.feature_block_config['cnn_setting']['padding_value'] = args.padding_value
    builder.feature_block_config['bottleneck_factor'] = args.bottleneck_factor
    builder.feature_block_config['compression_factor'] = args.compression_factor
    builder.feature_block_config['dropout'] = args.feature_dropout
    
    builder.relation_block_config['rnn_type'] = args.rnn_type
    builder.relation_block_config['rnn_setting']['num_layers'] = args.rnn_num
    builder.relation_block_config['rnn_setting']['hidden_size'] = args.rnn_size
    builder.relation_block_config['rnn_setting']['train_init_value'] = args.train_init_value
    builder.relation_block_config['rnn_setting']['customized_gru_init_mode'] = args.customized_gru_init_mode
    builder.relation_block_config['rnn_setting']['dropout'] = args.rnn_dropout
    
    builder.project_layer_config['kernel_size'] = args.project_kernel_size
    
    builder.discrim_config['rnn_num'] = args.disrim_rnn_num
    builder.discrim_config['rnn_size'] = args.disrim_rnn_size
    builder.discrim_config['train_init_value'] = args.train_init_value
    builder.use_discrim = args.use_discrim

    builder.out_channels = args.out_channels
    builder.use_sigmoid = args.use_sigmoid
    builder.site_ann_method = args.site_ann_method
    builder.predict_site_by = args.predict_site_by
        
    with open(args.config_path,"w") as fp:
        json.dump(builder.config, fp, indent=4)
