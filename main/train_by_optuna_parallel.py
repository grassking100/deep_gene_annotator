import os
import sys
import torch
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import write_json,create_folder
from sequence_annotation.utils.process import Process,process_schedule
from sequence_annotation.process.optuna import create_study
   
COMMAND = "python3 "+os.path.dirname(__file__)+"/train_by_optuna.py -s {} -t {} -v {} -e {} -b {} -n {} -i {}"
    
def get_n_completed_trial(study):
    df = study.trials_dataframe()
    if df.empty:
        return 0
    else:
        return sum(df['state'] == "COMPLETE")
    
def main(saved_root,train_data_path,val_data_path,epoch,batch_size,n_initial_points,n_trials,
         gpu_ids,is_maximize,by_grid_search,trial_start_number):

    batch_status = None
    optimized_status = None
    batch_status_path = os.path.join(saved_root,"batch_status.tsv")
    optimized_status_path = os.path.join(saved_root,"optimized_status.tsv")

    if os.path.exists(batch_status_path):
        batch_status = pd.read_csv(batch_status_path,sep='\t')
        
    if os.path.exists(optimized_status_path):    
        optimized_status = pd.read_csv(optimized_status_path,sep='\t')
    
    #Create folder
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)
    #Save setting
    setting_path = os.path.join(saved_root,"parallel_main_setting.json")
    write_json(config,setting_path)

    command = COMMAND.format(saved_root,train_data_path,val_data_path,epoch,
                             batch_size,1,n_initial_points)
    direction = 'minimize'
    if is_maximize:
        command += ' --is_maximize'
        direction = 'maximize'
        
    if by_grid_search:
        command += ' --by_grid_search'
        
    command += ' --trial_start_number {}'.format(trial_start_number) 
    
    study = create_study(saved_root,direction=direction,load_if_exists=True)
    n_completed = get_n_completed_trial(study)
    n_total = n_completed + n_trials
    n_optimized = n_total - n_initial_points
    n_random = n_initial_points - n_completed
    if n_random > 0:
        processes = []
        for index in range(n_random):
            process = Process(command)
            processes.append(process)

        status = process_schedule(processes,gpu_ids)
        if batch_status is None:
            batch_status = status
        else:
            batch_status = batch_status.append(status)
        batch_status.to_csv(batch_status_path,sep='\t',index=False)
        
        n_trial = get_n_completed_trial(study)-n_completed
        if n_random != n_trial:  
            raise Exception("Some trials are failed")

    if n_optimized >0:
        command = COMMAND.format(saved_root,train_data_path,val_data_path,epoch,
                                 batch_size,n_optimized,n_initial_points)
        if is_maximize:
            command += ' --is_maximize'
            
        command += ' --trial_start_number {}'.format(trial_start_number) 
            
        processes = [Process(command)]
        status = process_schedule(processes,gpu_ids)
        if optimized_status is None:
            optimized_status = status
        else:
            optimized_status = optimized_status.append(status)
        optimized_status.to_csv(optimized_status_path,sep='\t',index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("-e","--epoch",type=int,default=100)
    parser.add_argument("-b","--batch_size",type=int,default=32)
    parser.add_argument("-n","--n_trials",type=int,default=1)   
    parser.add_argument("-i","--n_initial_points",type=int,default=0)
    parser.add_argument("-g","--gpu_ids",type=lambda x: int(x).split(','),
                        default=list(range(torch.cuda.device_count())),help="GPUs to used")
    parser.add_argument("--is_maximize",action='store_true')
    parser.add_argument("--by_grid_search",action='store_true')
    parser.add_argument("--trial_start_number",type=int,default=0)

    args = parser.parse_args()
    config = dict(vars(args))
    main(**config)
