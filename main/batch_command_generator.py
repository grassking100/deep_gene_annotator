import os,sys
import pandas as pd
from argparse import ArgumentParser

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-d","--data_usage_table_path",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-m","--model_config_path",help="Path of Model config",required=True)
    parser.add_argument("-n","--save_result_name",required=True)
    parser.add_argument("-s","--save_command_table_path",required=True)
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
        saved_roots.append(item['root'])
        train_paths.append(os.path.join(item['root'],item['train_path']))
        val_paths.append(os.path.join(item['root'],item['val_path']))
        
    for paths in zip(train_paths,val_paths,saved_roots):
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
