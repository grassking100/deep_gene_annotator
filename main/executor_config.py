import sys,os
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import write_json,get_time_str

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_path",help="Path to save config",required=True)
    parser.add_argument("--use_native",action="store_true")
    parser.add_argument("--intron_coef",type=float,default=1)
    parser.add_argument("--other_coef",type=float,default=1)
    parser.add_argument("--gamma",type=int,default=0)
    parser.add_argument("--learning_rate",type=float)
    parser.add_argument("--weight_decay",type=float,default=0)
    parser.add_argument("--clip_grad_value",type=float)
    parser.add_argument("--clip_grad_norm",type=float)
    parser.add_argument("--grad_norm_type",type=float)
    parser.add_argument("--optim_type",type=str)
    parser.add_argument("--momentum",type=float)
    parser.add_argument("--target_weight_decay",type=lambda x: [float(v) for v in x.split(',')])
    parser.add_argument("--weight_decay_name",type=lambda x:x.split(','))
    parser.add_argument("--nesterov",action='store_true')
    parser.add_argument("--reduce_lr_on_plateau",action='store_true')
    parser.add_argument("--amsgrad",action='store_true')
    parser.add_argument("--adam_betas",type=lambda x: [float(v) for v in x.split(',')])

    args = parser.parse_args()
    kwargs = vars(args)
    config = {}
    loss_config = {}
    optim_config = {}
    weight_decay_config = {}
    loss_config_keys = ['intron_coef','other_coef','gamma']
    optim_config_keys = ['learning_rate','weight_decay','clip_grad_value','clip_grad_norm','grad_norm_type',
                         'optim_type','momentum','nesterov','reduce_lr_on_plateau','amsgrad','adam_betas']

    for key in loss_config_keys:
        loss_config[key] = kwargs[key]
     
    for key in optim_config_keys:
        optim_config[key] = kwargs[key]
    
    for key in ['target_weight_decay','weight_decay_name']:
        weight_decay_config[key] = kwargs[key]
    
    config['use_native'] = args.use_native
    config['loss_config'] = loss_config
    config['optim_config'] = optim_config
    config['weight_decay_config'] = weight_decay_config
    config['generated_time'] = get_time_str()
    
    write_json(config,args.config_path)
