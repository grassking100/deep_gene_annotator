import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_json,get_time_str

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-o","--output_path",help="Path to save settings",required=True)
    #CNN settings
    parser.add_argument("--customized_cnn_init")
    parser.add_argument("--cnn_act")
    parser.add_argument("--cnn_hidden",type=int)
    parser.add_argument("--cnn_kernel",type=int)
    parser.add_argument("--cnn_num",type=int)
    parser.add_argument("--feature_dropout",type=float)
    parser.add_argument("--norm_mode")
    #Relation settings
    parser.add_argument("--rnn_hidden",type=int)
    parser.add_argument("--rnn_num",type=int)
    parser.add_argument("--customized_rnn_init")
    parser.add_argument("--customized_relation_cnn_init")
    parser.add_argument("--relation_dropout",type=float)
    parser.add_argument("--out_channels",type=int)
    parser.add_argument("--output_act")
    #Norm settings
    parser.add_argument("--norm_class")
    parser.add_argument("--norm_momentum",type=float)
    parser.add_argument("--norm_affine",action='store_true')

    args = parser.parse_args()
    kwargs = vars(args)
    settings = {}
    feature_block_settings = {}
    relation_block_settings = {}
    norm_settings = {}
    
    #Feature settings
    feature_block_settings['customized_init'] = args.customized_cnn_init
    feature_block_settings['num_layers'] = args.cnn_num
    feature_block_settings['hidden_size'] = args.cnn_hidden
    feature_block_settings['kernel_size'] = args.cnn_kernel
    feature_block_settings['act_func'] = args.cnn_act
    feature_block_settings['dropout'] = args.feature_dropout
    feature_block_settings['norm_mode'] = args.norm_mode
    
    #Relation settings
    relation_block_settings['num_layers'] = args.rnn_num
    relation_block_settings['hidden_size'] = args.rnn_hidden
    relation_block_settings['customized_rnn_init'] = args.customized_rnn_init
    relation_block_settings['customized_cnn_init'] = args.customized_relation_cnn_init
    relation_block_settings['dropout'] = args.relation_dropout
    relation_block_settings['output_act'] = args.output_act
    relation_block_settings['out_channels'] = args.out_channels
    
    #Norm settings
    norm_settings['norm_class'] = args.norm_class
    norm_settings['momentum'] = args.norm_momentum
    norm_settings['affine'] = args.norm_affine

    settings['feature_block_kwargs'] = feature_block_settings
    settings['relation_block_kwargs'] = relation_block_settings
    settings['norm_kwargs'] = norm_settings
    settings['generated_time'] = get_time_str()
    
    write_json(settings,args.output_path)
