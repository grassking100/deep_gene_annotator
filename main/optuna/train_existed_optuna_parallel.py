import os
import sys
import torch
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.process_schedule import Process,process_schedule
   
COMMAND = "python3 "+os.path.dirname(__file__)+"/train_existed_optuna.py -i {} -n {}"

def main(root,gpu_ids,force=False):
    path = os.path.join(root,'trial_list.tsv')
    if not os.path.exists(path):
        raise Exception("the {} is not exist".format(path))
        
    batch_status = pd.read_csv(path,sep='\t')
    paths = list(batch_status['path'])
    processes = []
    for path in paths:
        command = COMMAND.format(root,path)
        if force:
            command += " -f"
        process = Process(command)
        processes.append(process)

    process_schedule(processes,gpu_ids)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i","--root",help="Root of files generated by train_by_optuna_parallel.py",required=True)
    parser.add_argument("-g","--gpu_ids",type=lambda x: [int(item) for item in x.split(',')],
                        default=list(range(torch.cuda.device_count())),help="GPUs to used")
    parser.add_argument("-f","--force",action='store_true')

    args = parser.parse_args()
    config = dict(vars(args))
    
    main(**config)
