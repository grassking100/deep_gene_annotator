import os
import pandas as pd
from argparse import ArgumentParser

def _get_name(path):
    return path.split('/')[-1].split('.')[0]

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-d","--data_usage_table_path",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-m","--model_config_path",help="Path of Model config",required=True)
    parser.add_argument("-n","--save_result_name",required=True)
    parser.add_argument("-s","--saved_root",required=True)
    parser.add_argument("-c","--save_command_table_path",required=True)
    parser.add_argument("--mode",default='w')

    args = parser.parse_args()
    
    command = "python3 {} -m {} -e {} -t {} -v {} -s {}/{} --batch_size 8 --patient 20 --epoch 100 --period 1"
    PROJECT_PATH=os.path.dirname(os.path.abspath(__file__+"/.."))
    MAIN_PATH = '{}/main/train_model.py'.format(PROJECT_PATH)
    
    for path in [MAIN_PATH,args.model_config_path,args.executor_config_path]:
        if not os.path.exists(path):
            raise Exception("{} is not exists".format(path))
    
    saved_roots = []
    train_paths = []
    val_paths = []
    data_usage = pd.read_csv(args.data_usage_table_path,comment='#').to_dict('record')
    for item in data_usage:
        name = "{}_{}".format(_get_name(item['training_path']),
                              _get_name(item['validation_path']))
        saved_root = os.path.join(args.saved_root,name)
        saved_roots.append(saved_root)
        train_paths.append(item['training_path'])
        val_paths.append(item['validation_path'])
        
    for paths in zip(train_paths,val_paths):
        if not all([os.path.exists(p) for p in paths]):
            raise Exception("Some paths, {}, are not exists".format(paths))
        
    with open(args.save_command_table_path,args.mode) as fp:
        if args.mode == 'w':
            fp.write("command\n")
            
        for paths in zip(train_paths,val_paths,saved_roots):
            train_path,val_path,saved_root = paths
            command_ = command.format(MAIN_PATH,args.model_config_path,
                                      args.executor_config_path,
                                      train_path,val_path,saved_root,
                                      args.save_result_name)
            command_ += " -g {}"
            fp.write(command_)
            fp.write("\n")
