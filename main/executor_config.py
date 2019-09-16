import json
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_path",help="Path to save config",required=True)
    parser.add_argument("--use_naive",action="store_true")
    parser.add_argument("--use_discrim",action="store_true")
    parser.add_argument("--learning_rate",type=float,default=1e-3)
    parser.add_argument("--disrim_learning_rate",type=float,default=1e-3)
    parser.add_argument("--intron_coef",type=float,default=1)
    parser.add_argument("--other_coef",type=float,default=1)
    parser.add_argument("--nontranscript_coef",type=float,default=0)
    parser.add_argument("--gamma",type=int,default=0)
    parser.add_argument("--transcript_output_mask",action="store_true")
    parser.add_argument("--transcript_answer_mask",action="store_true")
    parser.add_argument("--mean_by_mask",action="store_true")
    
    args = parser.parse_args()    
    config_path = args.config_path
    config = vars(args)
    del config['config_path']
    
    with open(config_path,"w") as fp:
        json.dump(config, fp, indent=4)
