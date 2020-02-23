import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import write_json,copy_path,create_folder,read_json
from sequence_annotation.process.optuna import OptunaTrainer
from main.train_model import train
from main.utils import backend_deterministic,load_data
from main.model_executor_creator import ModelExecutorCreator

def main(optuna_root,saved_root,epoch=None):
    saved_name = saved_root.split('/')[-1]
    trial_settings = read_json(os.path.join(saved_root,'{}_config.json'.format(saved_name)))
    optuna_settings = read_json(os.path.join(optuna_root,'parallel_main_setting.json'))
    train_data_path = optuna_settings['train_data_path']
    val_data_path = optuna_settings['val_data_path']
    if epoch is None:
        epoch = optuna_settings['epoch']
    batch_size = optuna_settings['batch_size']
    is_maximize = optuna_settings['is_maximize']
    trial_number = trial_settings['number']
    
    backend_deterministic(False)
    creator = ModelExecutorCreator()
    #Load, parse and save data
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)

    if is_maximize:
        monitor_target = 'val_macro_F1'
    else:
        monitor_target = 'val_loss'
    
    trainer = OptunaTrainer(train,optuna_root,creator,train_data,val_data,epoch,batch_size,
                            monitor_target=monitor_target,is_minimize=not is_maximize)
    trainer.train_single_trial(trial_number)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-r","--optuna_root",help="Root of optuna optimzation files",required=True)
    parser.add_argument("-s","--saved_root",help="Root of file to resume trainging",required=True)
    parser.add_argument("-e","--epoch",type=int,default=None)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")

    args = parser.parse_args()
    config = dict(vars(args))

    del config['gpu_id']    
    with torch.cuda.device(args.gpu_id):
        main(**config)
