import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import read_json
from sequence_annotation.process.optuna import OptunaTrainer
from sequence_annotation.genome_handler.select_data import load_data
from main.train_model import train
from main.utils import backend_deterministic
from main.model_executor_creator import ModelExecutorCreator

def main(optuna_root,saved_root,epoch=None):
    saved_name = saved_root.split('/')[-1]
    config_path = os.path.join(saved_root,'{}_config.json'.format(saved_name))
    print("Use existed config path, {}, to build trainer".format(config_path))
    trial_settings = read_json(config_path)
    optuna_settings = read_json(os.path.join(optuna_root,'optuna_setting.json'))
    train_data_path = optuna_settings['train_data_path']
    val_data_path = optuna_settings['val_data_path']
    if epoch is None:
        epoch = optuna_settings['epoch']
    batch_size = optuna_settings['batch_size']
    is_maximize = optuna_settings['is_maximize']
    augment_up_max = optuna_settings['augment_up_max']
    augment_down_max = optuna_settings['augment_down_max']
    discard_ratio_max = optuna_settings['discard_ratio_max']
    discard_ratio_min = optuna_settings['discard_ratio_min']
    has_cnn = optuna_settings['has_cnn']
    clip_grad_norm = optuna_settings['clip_grad_norm']
    grad_norm_type = optuna_settings['grad_norm_type']
    save_distribution = optuna_settings['save_distribution']
    use_lr_scheduler = optuna_settings['use_lr_scheduler']
    

    backend_deterministic(False)
    creator = ModelExecutorCreator(clip_grad_norm=clip_grad_norm,
                                   grad_norm_type=grad_norm_type,
                                   has_cnn=has_cnn,
                                   use_lr_scheduler=use_lr_scheduler)
    #Load, parse and save data
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)

    if is_maximize:
        monitor_target = 'val_macro_F1'
    else:
        monitor_target = 'val_loss'
    
    trainer = OptunaTrainer(train,optuna_root,creator,train_data,val_data,epoch,batch_size,
                            monitor_target=monitor_target,is_minimize=not is_maximize,
                            discard_ratio_max=discard_ratio_max,discard_ratio_min=discard_ratio_min,
                            augment_up_max=augment_up_max,augment_down_max=augment_down_max,
                            save_distribution=save_distribution)

    trainer.train_single_trial(trial_settings)

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
