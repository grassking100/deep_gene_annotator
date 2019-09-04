import os
import sys
from argparse import ArgumentParser
import json
sys.path.append("/home/sequence_annotation")
from sequence_annotation.pytorch.model import SeqAnnBuilder

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("--config_path",help="Path to save config",required=True)
    parser.add_argument("--use_naive",action="store_true")
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
    args = parser.parse_args()

    builder = SeqAnnBuilder()
    builder.feature_block_config['num_layers'] = args.cnn_num
    builder.feature_block_config['cnn_setting']['out_channels'] = args.cnn_out
    builder.feature_block_config['cnn_setting']['kernel_size'] = args.cnn_kernel
    builder.feature_block_config['cnn_setting']['cnn_act'] = args.cnn_act
    builder.feature_block_config['stack_cnn_class'] = args.stack_cnn_class
    
    builder.relation_block_config['rnn_setting']['num_layers'] = args.rnn_num
    builder.relation_block_config['rnn_setting']['hidden_size'] = args.rnn_size
    builder.relation_block_config['rnn_setting']['train_init_value'] = args.train_init_value
    
    builder.relation_block_config['rnn_type'] = args.rnn_type
    builder.discrim_config['rnn_num'] = args.disrim_rnn_num
    builder.discrim_config['rnn_size'] = args.disrim_rnn_size
    builder.discrim_config['train_init_value'] = args.train_init_value
    builder.use_discrim = args.use_discrim
    if args.use_naive:
        builder.out_channels = 3
        builder.use_sigmoid = False
    else:    
        builder.out_channels = 2
        builder.use_sigmoid = True
    
    with open(args.config_path,"w") as fp:
        json.dump(builder.config, fp, indent=4)
