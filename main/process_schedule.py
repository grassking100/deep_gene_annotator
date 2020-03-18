import os,sys
import pandas as pd
import torch
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.process import Process,process_schedule
from sequence_annotation.utils.utils import create_folder

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-i","--cmd_table_path",required=True)
    parser.add_argument("-o","--output_path",required=True,
                        help="Path of saved result table (in tsv format)")
    parser.add_argument("-g","--gpu_ids",type=lambda x: [int(item) for item in x.split(',')],
                        default=list(range(torch.cuda.device_count())),help="GPUs to used")
    
    args = parser.parse_args()
    
    device_count = torch.cuda.device_count()
    if device_count < len(args.gpu_ids):
        raise Exception("Inconsist between max device number {} and required number {}".format(device_count,len(args.gpu_ids)))
    
    if os.path.exists(args.output_path):
        raise Exception("The output path, {}, is already exist".format(args.output_path))
    
    commands = list(pd.read_csv(args.cmd_table_path,comment='#')['command'])
    
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
