import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import write_json,copy_path,create_folder
from sequence_annotation.process.optuna import OptunaTrainer
from main.train_model import train
from main.utils import backend_deterministic,load_data
from main.model_executor_creator import ModelExecutorCreator

def main(saved_root,train_data_path,val_data_path,
         epoch,batch_size,n_initial_points,n_trials,is_maximize,**kwargs):

    backend_deterministic(False)
    creator = ModelExecutorCreator()
    write_json({'space size':creator.space_size},os.path.join(saved_root,'space_size.json'))
    #Load, parse and save data
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)
    copied_paths = [train_data_path,val_data_path]
    source_backup_path = os.path.join(saved_root,'source')
    create_folder(source_backup_path)
    for path in copied_paths:
        copy_path(source_backup_path,path)

    if is_maximize:
        monitor_target = 'val_macro_F1'
    else:
        monitor_target = 'val_loss'
    
    trainer = OptunaTrainer(train,saved_root,creator,train_data,val_data,epoch,batch_size,
                            monitor_target=monitor_target,is_minimize=not is_maximize,**kwargs)
    trainer.optimize(n_initial_points,n_trials)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("-g","--gpu_id",default=0,help="GPU to used",type=int)
    parser.add_argument("-e","--epoch",type=int,default=100)
    parser.add_argument("-b","--batch_size",type=int,default=32)
    parser.add_argument("-n","--n_trials",type=int,default=None)
    parser.add_argument("-i","--n_initial_points",type=int,default=None)
    parser.add_argument("--is_maximize",action='store_true')
    parser.add_argument("--by_grid_search",action='store_true')
    parser.add_argument("--trial_start_number",type=int,default=0)
    
    args = parser.parse_args()
    config = dict(vars(args))
    #Create folder
    if not os.path.exists(args.saved_root):
        os.mkdir(args.saved_root)
    #Save setting
    setting_path = os.path.join(args.saved_root,"optuna_setting.json")
    write_json(config,setting_path)

    del config['gpu_id']    
    with torch.cuda.device(args.gpu_id):
        main(**config)
