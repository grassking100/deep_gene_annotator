import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("-g","--gpu_id",default='0',help="GPU to used")
    parser.add_argument("--last_act")
    parser.add_argument("--use_cnn",action='store_true')
    parser.add_argument("--out_channels",type=int,default=3)
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--n_trials",type=int,default=10)   
    parser.add_argument("--set_pruner",action='store_true')
    parser.add_argument("--n_startup_trials",type=int,default=10)
    parser.add_argument("--n_warmup_steps",type=int,default=50)
    parser.add_argument("--rnn_type")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id

import sys
import json
import deepdish as dd
import optuna
from optuna.pruners import MedianPruner,NopPruner
from optuna.structs import TrialPruned
from optuna.integration import SkoptSampler
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import write_fasta, create_folder
from sequence_annotation.process.callback import OptunaPruning
from sequence_annotation.genome_handler.ann_genome_processor import simplify_genome
from main.utils import load_data, get_model, get_executor, copy_path, SIMPLIFY_MAP
from main.train_model import train
from main.space_generator import Builder
    
def evaluate_generator(data,executor_config,epoch=None,batch_size=None):
    def evaluate(saved_root,model_config,trial):
        if not os.path.exists(saved_root):
            os.mkdir(saved_root)
        path = os.path.join(saved_root,'trial_{}_model_config.json'.format(trial.number))
        with open(path,"w") as fp:
            json.dump(model_config,fp, indent=4)
        path = os.path.join(saved_root,'trial_{}_executor_config.json'.format(trial.number))
        with open(path,"w") as fp:
            json.dump(executor_config,fp, indent=4)
        path = os.path.join(saved_root,'trial_{}_params.json'.format(trial.number))
        with open(path,"w") as fp:
            json.dump(trial.params,fp, indent=4)
        train_data, val_data = data
        model = get_model(model_config)
        executor = get_executor(model,**executor_config)
        pruning = OptunaPruning(trial)
        other_callbacks = [pruning]
        worker = train(model,executor,train_data,val_data,saved_root,
                       epoch=epoch,batch_size=batch_size,period=1,
                       use_gffcompare=False,other_callbacks=other_callbacks)
        #Save best result as final result
        trial.report(worker.best_result['val_loss'])
        if pruning.is_prune:
            raise TrialPruned()
        return worker.best_result['val_loss']
    return evaluate

def objective_generator(saved_root,data,spcae_generator,executor_config,epoch=None,batch_size=None):
    evaluate = evaluate_generator(data,executor_config,epoch=epoch,batch_size=batch_size)
    def objective(trial):
        model_config = spcae_generator(trial)
        path = os.path.join(saved_root,"trial_{}").format(trial.number)
        result = evaluate(path,model_config,trial)
        return result
    return objective

if __name__ == '__main__':
    builder = Builder()
    cnn_num = None
    if not args.use_cnn:
        cnn_num = 0
    space_generator = builder.build(args.out_channels,args.last_act,
                                    cnn_num=cnn_num,rnn_type=args.rnn_type)

    args = parser.parse_args()
    config = vars(args)
    #Create folder
    if not os.path.exists(args.saved_root):
        os.mkdir(args.saved_root)
    #Save setting
    copy_path(args.saved_root,args.executor_config_path)
    setting_path = os.path.join(args.saved_root,"main_setting.json")
    with open(setting_path,"w") as fp:
        json.dump(config, fp, indent=4)

    #Load, parse and save data
    train_data = dd.io.load(args.train_data_path)
    val_data = dd.io.load(args.val_data_path)
    train_data = train_data[0],simplify_genome(train_data[1],SIMPLIFY_MAP)
    val_data = val_data[0],simplify_genome(val_data[1],SIMPLIFY_MAP)
    data = train_data, val_data
    write_fasta(os.path.join(args.saved_root,'train.fasta'),train_data[0])
    write_fasta(os.path.join(args.saved_root,'val.fasta'),val_data[0])    
    
    data_path = os.path.join(args.saved_root,"data.h5")
    if not os.path.exists(data_path):
        dd.io.save(data_path,data)

    #Create executor
    with open(args.executor_config_path,"r") as fp:
        executor_config = json.load(fp)

    #Create pruner
    if args.set_pruner:
        pruner = MedianPruner(n_startup_trials=args.n_startup_trials,
                              n_warmup_steps=args.n_warmup_steps)
    else:
        pruner = NopPruner()
        
    objective = objective_generator(args.saved_root,data,space_generator,executor_config,
                                    epoch=args.epoch,batch_size=args.batch_size)
    study = optuna.create_study(study_name='seq_ann',
                                storage='sqlite:///{}/trials.db'.format(args.saved_root),
                                load_if_exists=True,
                                pruner=pruner,
                                sampler=SkoptSampler())
    study.optimize(objective, n_trials=args.n_trials)
    