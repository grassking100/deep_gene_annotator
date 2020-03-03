import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.process.optuna import add_exist_trials,create_study

def _create_study(output_root,is_maximize):
    if is_maximize:
        direction = 'maximize'
    else:
        direction = 'minimize'
    study = create_study(output_root,load_if_exists=False,direction=direction)
    return study

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t","--trial_list_path",help="Path of trial list table",required=True)
    parser.add_argument("-o","--output_root",help="Root of create trial storage database",required=True)
    parser.add_argument("--is_maximize",action='store_true')
    parser.add_argument("--trial_start_number",type=int,default=0)
    
    args = parser.parse_args()
    study = _create_study(args.output_root,is_maximize=args.is_maximize)
    add_exist_trials(study,args.trial_list_path,args.trial_start_number)
