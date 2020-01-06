import os,sys
import pandas as pd
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from sequence_annotation.utils.process import Process,process_schedule
from sequence_annotation.utils.utils import create_folder

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("--cmd_table_path",required=True)
    parser.add_argument("-o","--output_path",help="Root to save result table (in tsv format)",required=True)
    parser.add_argument("-g","--gpu_ids",type=lambda x: int(x).split(','),
                        default=list(range(torch.cuda.device_count())),help="GPUs to used")
    
    args = parser.parse_args()
    
    if torch.cuda.device_count() != len(args.gpu_ids):
        raise Exception("Inconsist GPU number")
    
    commands = list(pd.read_csv(args.cmd_table_path)['command'])
    
    root = args.output_path.split('/')[:-1]
    if len(root)>=0:
        root = '/'.join(root)
        create_folder(root)
    
    processes = []
    for index,command in enumerate(commands):
        process = Process(command,name=index)
        processes.append(process)
        
    status = process_schedule(processes,args.gpu_ids)
    status.to_csv(args.output_path,sep='\t',index=False)
