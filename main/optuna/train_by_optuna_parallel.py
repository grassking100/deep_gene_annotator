import os
import sys
import torch
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_json,read_json
from sequence_annotation.utils.process_schedule import Process,process_schedule
from sequence_annotation.process.optuna import create_study
from sequence_annotation.visual.save_optuna_image import main as save_optuna_image_main
   
COMMAND = "python3 "+os.path.dirname(__file__)+"/train_by_optuna.py -s {} -t {} -v {} -e {} -b {} -n {} -i {}"
    
def get_n_completed_trial(study):
    df = study.trials_dataframe()
    if df.empty:
        return 0
    else:
        return sum(df['state'] == "COMPLETE")
    
def _append_command(command,appended_command=None,is_maximize=False):
    if is_maximize:
        command += ' --is_maximize'

    if appended_command is not None:
        command += ' ' + appended_command

    return command
    
def main(output_root,train_data_path,val_data_path,epoch,batch_size,
         n_startup_trials,n_total,gpu_ids,patient_trial_num=None,
         appended_command=None,is_maximize=False):

    batch_status = None
    optimized_status = None
    batch_status_path = os.path.join(output_root,"batch_status.tsv")
    optimized_status_path = os.path.join(output_root,"optimized_status.tsv")

    if os.path.exists(batch_status_path):
        batch_status = pd.read_csv(batch_status_path,sep='\t')
        
    if os.path.exists(optimized_status_path):    
        optimized_status = pd.read_csv(optimized_status_path,sep='\t')
    
    #Create folder
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    #Save setting
    setting_path = os.path.join(output_root,"parallel_main_setting.json")
    if os.path.exists(setting_path):
        existed = dict(read_json(setting_path))
        config_ = dict(config)
        del existed['gpu_ids']
        del config_['gpu_ids']
        if config_ != existed:
            raise Exception("The {} is not same as previous one".format(setting_path))
    else:
        write_json(config,setting_path)

    command = COMMAND.format(output_root,train_data_path,val_data_path,epoch,
                             batch_size,1,n_startup_trials)
    
    command = _append_command(command,appended_command=appended_command,is_maximize=is_maximize)
    
    direction = 'maximize' if is_maximize else 'minimize'
    study = create_study(output_root,direction=direction,load_if_exists=True)
    n_completed = get_n_completed_trial(study)
    if n_completed>=n_startup_trials:
        n_random = 0
        n_optimized = n_total-n_completed
    else:
        n_random = n_startup_trials - n_completed
        n_optimized = n_total-n_startup_trials
        
    print("Number of complete is {}".format(n_completed))
    print("Number of startup trials is {}".format(n_startup_trials))
    print("Number of random is {}".format(n_random))
    print("Number of optimized is {}".format(n_optimized))
        
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
        print("Start optimization")
        if patient_trial_num is not None:
            waited_num = 0
            if len(study.trials) != n_startup_trials:
                for index in range(n_startup_trials,len(study.trials)):
                    if index == study.best_trial.number:
                        waited_num = 0
                    else:
                        waited_num += 1
            while True:
                print("Waited number: {}".format(waited_num))
                if waited_num >= patient_trial_num or len(study.trials)==n_total:
                    break
                command = COMMAND.format(output_root,train_data_path,val_data_path,epoch,
                                         batch_size,1,n_startup_trials)

                command = _append_command(command,appended_command=appended_command,
                                          is_maximize=is_maximize)

                processes = [Process(command)]
                status = process_schedule(processes,gpu_ids)
                if optimized_status is None:
                    optimized_status = status
                else:
                    optimized_status = optimized_status.append(status)
                if len(study.trials)-1 == study.best_trial.number:
                    waited_num = 0
                else:
                    waited_num += 1

        else:
            command = COMMAND.format(output_root,train_data_path,val_data_path,epoch,
                                     batch_size,n_optimized,n_startup_trials)

            command = _append_command(command,appended_command=appended_command,
                                      is_maximize=is_maximize)

            processes = [Process(command)]
            status = process_schedule(processes,gpu_ids)

            if optimized_status is None:
                optimized_status = status
            else:
                optimized_status = optimized_status.append(status)
        optimized_status.to_csv(optimized_status_path,sep='\t',index=False)
        save_optuna_image_main(output_root, os.path.join(output_root,'stats'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("-o","--output_root",help="Root to save file",required=True)
    parser.add_argument("-e","--epoch",type=int,default=100)
    parser.add_argument("-b","--batch_size",type=int,default=32)
    parser.add_argument("-n","--n_total",type=int,default=1)   
    parser.add_argument("-i","--n_startup_trials",type=int,default=0)
    parser.add_argument("-g","--gpu_ids",type=lambda x: [int(item) for item in x.split(',')],
                        default=list(range(torch.cuda.device_count())),help="GPUs to used")
    parser.add_argument("--is_maximize",action='store_true')
    parser.add_argument("--appended_command",type=str,default=None)
    parser.add_argument("--patient_trial_num",type=int,default=None)
    

    args = parser.parse_args()
    config = dict(vars(args))
    main(**config)
