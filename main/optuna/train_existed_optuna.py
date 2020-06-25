import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_json
from sequence_annotation.process.optuna import OptunaTrainer
from sequence_annotation.genome_handler.select_data import load_data
from main.utils import backend_deterministic
from main.deep_learning.train_model import train
from main.optuna.model_executor_creator import ModelExecutorCreator

def main(optuna_root,trial_name,force=False):
    output_root = os.path.join(optuna_root,trial_name)
    config_path = os.path.join(output_root,'{}_config.json'.format(trial_name))
    print("Use existed config path, {}, to build trainer".format(config_path))
    trial_settings = read_json(config_path)
    optuna_settings = read_json(os.path.join(optuna_root,'optuna_setting.json'))
    #clip_grad_norm = optuna_settings['clip_grad_norm']
    #grad_norm_type = optuna_settings['grad_norm_type']
    #has_cnn = optuna_settings['has_cnn']
    #lr_scheduler_patience = optuna_settings['lr_scheduler_patience']

    backend_deterministic(False)
    creator = ModelExecutorCreator(**optuna_settings)
    #Load, parse and save data
    train_data_path = optuna_settings['train_data_path']
    val_data_path = optuna_settings['val_data_path']
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)

    is_maximize = optuna_settings['is_maximize']
    if is_maximize:
        monitor_target = 'val_macro_F1'
    else:
        monitor_target = 'val_loss'

    epoch = optuna_settings['epoch']
    batch_size = optuna_settings['batch_size']

    del optuna_settings['train_data_path']
    del optuna_settings['val_data_path']
    del optuna_settings['clip_grad_norm']
    del optuna_settings['grad_norm_type']
    del optuna_settings['has_cnn']
    del optuna_settings['lr_scheduler_patience']
    del optuna_settings['epoch']
    del optuna_settings['batch_size']    
    del optuna_settings['is_maximize']
    del optuna_settings['output_root']
    del optuna_settings['n_startup_trials']
    del optuna_settings['n_trials']
    del optuna_settings['gpu_id']
        
        
    trainer = OptunaTrainer(train,optuna_root,creator,train_data,
                            val_data,epoch,batch_size,monitor_target,
                            is_minimize=not is_maximize,**optuna_settings)

    trainer.train_single_trial(trial_settings,force=force)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i","--optuna_root",help="Root of optuna optimization files",required=True)
    parser.add_argument("-n","--trial_name",help="Name of trail to resume training",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("-f","--force",action='store_true')

    args = parser.parse_args()
    config = dict(vars(args))

    del config['gpu_id']    
    with torch.cuda.device(args.gpu_id):
        main(**config)
