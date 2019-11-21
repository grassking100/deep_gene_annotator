import os,sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import model_setting_generator, model_settings_generator
from main.utils import get_model
from sequence_annotation.process.utils import param_num

command = "python3 /home/sequence_annotation/main/model_config.py --out_channels 2 --padding_handle 'partial' --config_path {} --rnn_type HierAttenGRU --customized_cnn 'kaiming_uniform_cnn_init' --customized_gru_init 'orth_in_xav_bias_zero_gru_init' --customized_rnn_cnn 'xavier_uniform_cnn_init' --cnn_kernel 3 --cnn_out {} --cnn_num {} --rnn_size {} --rnn_num {}"

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("--config_path",help="Path to save config",required=True)
    parser.add_argument("--hidden_size_base",type=int,required=True)
    parser.add_argument("--cnn_num_base",type=int,required=True)
    parser.add_argument("--alpha",type=float,required=True)
    parser.add_argument("--beta",type=float,required=True)
    parser.add_argument("--rnn_size",type=int)
    parser.add_argument("--rnn_num",type=int,required=True)
    parser.add_argument("--prefix",default='b0',type=str)
    parser.add_argument("--phi",type=int,default=1)
    
    args = parser.parse_args()
    basic_setting = model_setting_generator(args.hidden_size_base,args.cnn_num_base,phi=args.phi,
                                            alpha=args.alpha,beta=args.beta)
    o,d = basic_setting
    rnn_size = args.rnn_size or o
    setting_name = '{}_ck3o{}n{}_rh{}n{}'.format(args.prefix,o,d,rnn_size,args.rnn_num)
    setting_path = os.path.join(args.config_path,setting_name)+'.json'
    command_ = command.format(setting_path,o,d,rnn_size,args.rnn_num)
    os.system(command_)
    setting_dict = {str([o,d]):(args.alpha,args.beta,args.phi,param_num(get_model(setting_path)))}
    setting_path = os.path.join(args.config_path,'{}_parameter_num.tsv'.format(setting_name))
    
    df = pd.DataFrame.from_dict(setting_dict,orient='index',columns=['alpha','beta','phi','size'])
    df = df.sort_values(by=['size'],ascending=False)
    df.to_csv(setting_path,sep='\t',index_label='o,d')
