import os,sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from sequence_annotation.utils.utils import model_setting_generator, model_settings_generator

train_main_path = 'python3 /home/sequence_annotation/main/train_model.py'
only_train_command = "{} -g {} -m {} -e {} -t {} -s {} --batch_size {} --patient {} --epoch {} {}&"
train_val_command = "{} -g {} -m {} -e {} -t {} -v {} -s {} --batch_size {} --patient {} --epoch {} {}&"

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config_root",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of valdation data")
    parser.add_argument("--patient",help="Dafault value is 5. If value is 'none', then model won't be stopped",
                        type=lambda x: int(x) if x != 'none' else None,default=5)
    parser.add_argument("-g","--gpu_ids",type=lambda x: str(x).split(','),
                        default=['0','1','2','3'],help="GPUs to used")
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--test_wtih_fix_boundary",action='store_true')
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--only_train",action='store_true')
    parser.add_argument("--period",default=None,type=int)
    
    args = parser.parse_args()
    gpu_ids = args.gpu_ids
    model_config_names = sorted(os.listdir(args.model_config_root))
    gpu_index = 0
    print(model_config_names)
    extra = ''
    if args.test_wtih_fix_boundary:
        extra += '--test_wtih_fix_boundary '

    if args.save_distribution:
        extra += '--save_distribution '
        
    if args.only_train:
        extra += '--only_train ' 
        
    if args.period is not None:
        extra += '--period {}'.format(args.period)

    for model_config_name in model_config_names:
        if model_config_name.endswith('json') and model_config_name.startswith('model'):
            
            folder_name = model_config_name[6:-5]
            mdeol_config_path = os.path.join(args.model_config_root,model_config_name)
            saved_path = os.path.join(args.saved_root,folder_name)
            if args.val_data_path is None:
                command_ = only_train_command.format(train_main_path,gpu_ids[gpu_index],
                                                     mdeol_config_path,args.executor_config_path,
                                                     args.train_data_path,saved_path,
                                                     args.batch_size,args.patient,args.epoch,extra)
            else:
                command_ = train_val_command.format(train_main_path,gpu_ids[gpu_index],
                                                    mdeol_config_path,args.executor_config_path,
                                                    args.train_data_path,args.val_data_path,saved_path,
                                                    args.batch_size,args.patient,args.epoch,extra)
            print(command_)
            os.system(command_)
            gpu_index += 1
            gpu_index = gpu_index%len(gpu_ids)
