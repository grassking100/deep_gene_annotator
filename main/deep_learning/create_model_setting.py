import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_json,get_time_str

def main(output_path,customized_cnn_init=None,cnn_act=None,cnn_hidden=None,cnn_kernel=None,cnn_num=None,
         feature_dropout=None,norm_mode=None,rnn_hidden=None,rnn_num=None,customized_rnn_init=None,
         customized_relation_cnn_init=None,relation_dropout=None,out_channels=None,output_act=None,
         norm_class=None,norm_momentum=None,norm_affine=False):
    
    settings = {}
    feature_block_settings = {}
    relation_block_settings = {}
    norm_settings = {}
    
    #Feature settings
    feature_block_settings['customized_init'] = customized_cnn_init
    feature_block_settings['num_layers'] = cnn_num
    feature_block_settings['hidden_size'] = cnn_hidden
    feature_block_settings['kernel_size'] = cnn_kernel
    feature_block_settings['act_func'] = cnn_act
    feature_block_settings['dropout'] = feature_dropout
    feature_block_settings['norm_mode'] = norm_mode
    
    #Relation settings
    relation_block_settings['num_layers'] = rnn_num
    relation_block_settings['hidden_size'] = rnn_hidden
    relation_block_settings['customized_rnn_init'] = customized_rnn_init
    relation_block_settings['customized_cnn_init'] = customized_relation_cnn_init
    relation_block_settings['dropout'] = relation_dropout
    relation_block_settings['output_act'] = output_act
    relation_block_settings['out_channels'] = out_channels
    
    #Norm settings
    norm_settings['norm_class'] = norm_class
    norm_settings['momentum'] = norm_momentum
    norm_settings['affine'] = norm_affine

    settings['feature_block_kwargs'] = feature_block_settings
    settings['relation_block_kwargs'] = relation_block_settings
    settings['norm_kwargs'] = norm_settings
    settings['generated_time'] = get_time_str()
    
    write_json(settings,output_path)


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
    main(**kwargs)
    