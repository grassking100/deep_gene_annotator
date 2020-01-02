import sys
import json
from argparse import ArgumentParser
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import write_json

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_path",help="Path to save config",required=True)
    parser.add_argument("--use_native",action="store_true")
    parser.add_argument("--learning_rate",type=float)
    parser.add_argument("--intron_coef",type=float,default=1)
    parser.add_argument("--other_coef",type=float,default=1)
    parser.add_argument("--nontranscript_coef",type=float,default=0)
    parser.add_argument("--gamma",type=int,default=0)
    parser.add_argument("--transcript_output_mask",action="store_true")
    parser.add_argument("--transcript_answer_mask",action="store_true")
    parser.add_argument("--mean_by_mask",action="store_true")
    parser.add_argument("--weight_decay",type=float)
    parser.add_argument("--label_num",type=int)
    parser.add_argument("--predict_label_num",type=int)
    parser.add_argument("--answer_label_num",type=int)
    parser.add_argument("--output_label_num",type=int)
    parser.add_argument("--grad_clip",type=float)
    parser.add_argument("--grad_norm",type=float)
    parser.add_argument("--optim_type",type=str,required=True)
    parser.add_argument("--momentum",type=float)
    parser.add_argument("--target_weight_decay",type=lambda x: [float(v) for v in x.split(',')])
    parser.add_argument("--weight_decay_name",type=lambda x:x.split(','))
    parser.add_argument("--nesterov",action='store_true')
    parser.add_argument("--reduce_lr_on_plateau",action='store_true')
    parser.add_argument("--amsgrad",action='store_true')
    parser.add_argument("--adam_betas",type=lambda x: [float(v) for v in x.split(',')])

    args = parser.parse_args()    
    config_path = args.config_path
    config = vars(args)
    del config['config_path']
    
    write_json(config,config_path)
