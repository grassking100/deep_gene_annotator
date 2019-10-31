import os,sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import model_setting_generator, model_settings_generator

command = "python3 /home/sequence_annotation/main/model_config.py --out_channels 2 --use_sigmoid --padding_handle 'partial' --config_path {} --rnn_type HierAttenGRU --project_kernel_size 0 --customized_cnn 'kaming_uniform_cnn_init' --customized_gru_init '{}' --customized_rnn_cnn 'kaming_uniform_cnn_init' --cnn_kernel {} --cnn_out {} --rnn_size {} --cnn_num {} --rnn_num 4"

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("--config_path",help="Path to save config",required=True)
    parser.add_argument("-r","--raidus_base",type=int,required=True)
    parser.add_argument("-o","--hidden_size_base",type=int,required=True)
    parser.add_argument("-d","--cnn_num_base",type=int,required=True)
    parser.add_argument("--step",type=float,default=None)
    parser.add_argument("--min_value",type=float,default=None)
    parser.add_argument("--max_coef",type=float,default=None)
    parser.add_argument("--prefix",default='b0',type=str)
    parser.add_argument("--customized_gru_init",default='orth_in_xav_bias_zero_gru_init',type=str)
    
    args = parser.parse_args()
    settings = model_settings_generator(args.raidus_base,args.hidden_size_base,args.cnn_num_base,
                                        step=args.step,min_value=args.min_value)
    basic_setting = model_setting_generator(args.raidus_base,args.hidden_size_base,args.cnn_num_base)
    settings[basic_setting]=(1,1,1)
    for (l,o,d) in settings:
        setting_name = '{}_{}_{}_{}.json'.format(args.prefix,l,o,d)
        setting_path = os.path.join(args.config_path,setting_name)
        command_ = command.format(setting_path,args.customized_gru_init,l,o,o,d)
        os.system(command_)
        
    setting_path = os.path.join(args.config_path,'{}_setting_parameter.tsv'.format(args.prefix))
    pd.DataFrame.from_dict(settings,orient='index',columns=['alpha','beta','gamma']).to_csv(setting_path,sep='\t')
