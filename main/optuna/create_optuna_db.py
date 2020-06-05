import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.process.optuna import add_exist_trials,create_study

def _create_study(output_root,is_maximize):
    if is_maximize:
        direction = 'maximize'
    else:
        direction = 'minimize'
    study = create_study(output_root,load_if_exists=False,direction=direction)
    return study

def main(input_root,is_maximize,trial_start_number):
    trial_list_path = os.path.join(input_root,'trial_list.tsv')
    study = _create_study(input_root,is_maximize=is_maximize)
    add_exist_trials(study,trial_list_path,trial_start_number)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i","--input_root",help="Root of create trial storage database",required=True)
    parser.add_argument("-x","--is_maximize",action='store_true')
    parser.add_argument("--trial_start_number",type=int,default=0)
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
