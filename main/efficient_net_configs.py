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
    parser.add_argument("-o","--hidden_size_base",type=int,required=True)
    parser.add_argument("-d","--cnn_num_base",type=int,required=True)
    parser.add_argument("-x","--rnn_size",type=int)
    parser.add_argument("-n","--rnn_num",type=int,required=True)
    parser.add_argument("--step",type=float)
    parser.add_argument("--min_value",type=float)
    parser.add_argument("--max_coef",type=float)
    parser.add_argument("--prefix",default='b0',type=str)
    parser.add_argument("--add_basic",action='store_true')
    parser.add_argument("--remove_basic",action='store_true')
    parser.add_argument("--remove_one",action='store_true')
    parser.add_argument("--phi",type=int,default=1)
    
    args = parser.parse_args()
    settings = model_settings_generator(args.hidden_size_base,args.cnn_num_base,
                                        step=args.step,min_value=args.min_value,
                                        max_coef=args.max_coef,phi=args.phi,
                                        remove_one=args.remove_one)
    basic_setting = model_setting_generator(args.hidden_size_base,args.cnn_num_base)
    if args.remove_basic and basic_setting in settings:
        del settings[basic_setting]

    if args.add_basic:
        settings[basic_setting]=[1,1,0]

    setting_dict = {}
    for (o,d),(alpha,beta,phi) in settings.items():
        if args.rnn_size is not None:
            rnn_size = args.rnn_size
        else:
            rnn_size = o
        setting_name = '{}_ck3o{}n{}_rh{}n{}.json'.format(args.prefix,o,d,rnn_size,args.rnn_num)
        setting_path = os.path.join(args.config_path,setting_name)
        command_ = command.format(setting_path,o,d,rnn_size,args.rnn_num)
        os.system(command_)
        setting_dict[str([o,d])] = (alpha,beta,phi,param_num(get_model(setting_path)))
        
    setting_path = os.path.join(args.config_path,'{}_setting_parameter.tsv'.format(args.prefix))
    
    df = pd.DataFrame.from_dict(setting_dict,orient='index',columns=['alpha','beta','phi','size'])
    df = df.sort_values(by=['size'],ascending=False)
    df.to_csv(setting_path,sep='\t',index_label='o,d')
