import os
import sys
import math
import json
import deepdish as dd
from argparse import ArgumentParser
import optuna
from optuna.integration import SkoptSampler
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner,NopPruner
from optuna.structs import TrialPruned

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("-g","--gpu_id",type=str,default=0,help="GPU to used")
    parser.add_argument("--n_trials",type=int,default=10)
    parser.add_argument("--use_common_atten",action='store_true')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import write_fasta, create_folder
from sequence_annotation.process.callback import OptunaPruning
from sequence_annotation.genome_handler.ann_genome_processor import simplify_genome
from sequence_annotation.process.model import SeqAnnBuilder
from main.utils import load_data, get_model, get_executor, copy_path, SIMPLIFY_MAP
from main.train_model import train
    
KERNEL_BASE = 4
OUT_BASE = 4
DEPTH_BASE = 4
    
def HAGRU_space_generator(trial,padding_handle=None,use_common_atten=False):
    builder = SeqAnnBuilder()
    builder.relation_block_config['rnn_type'] = 'HierAttenGRU'
    builder.feature_block_config['cnn_setting']['padding_handle'] = padding_handle or 'partial'
    builder.feature_block_config['cnn_setting']['customized_init'] = 'kaming_uniform_cnn_init'
    builder.relation_block_config['rnn_setting']['customized_gru_init'] = 'orth_in_xav_bias_zero_gru_init'
    builder.relation_block_config['rnn_setting']['customized_cnn_init'] = 'kaming_uniform_cnn_init'
    
    COEF_MAX=2
    alpha = trial.suggest_uniform('alpha',1,COEF_MAX)
    beta_square = trial.suggest_uniform('beta_square',1,COEF_MAX)
    beta = math.pow(beta_square,0.5)
    gamma = math.pow(COEF_MAX/(alpha*beta_square),0.5)
    builder.feature_block_config['cnn_setting']['kernel_size'] = int(round(alpha*KERNEL_BASE)*2+1)
    builder.relation_block_config['rnn_setting']['num_layers'] = int(round(DEPTH_BASE))
    builder.feature_block_config['cnn_setting']['out_channels'] = int(round(beta*OUT_BASE))
    builder.relation_block_config['rnn_setting']['hidden_size'] = int(round(beta*OUT_BASE))
    builder.feature_block_config['num_layers'] = int(round(gamma*DEPTH_BASE))
    builder.relation_block_config['rnn_setting']['use_common_atten'] = use_common_atten
    builder.project_layer_config['kernel_size'] = 0
    return builder.config
    
def evaluate_generator(data,executor_config):
    def evaluate(saved_root,model_config,trial):
        epoch=100
        batch_size=100
        if not os.path.exists(saved_root):
            os.mkdir(saved_root)
        train_data, val_data = data
        model = get_model(model_config)
        executor = get_executor(model,**executor_config)
        pruning = OptunaPruning(trial)
        other_callbacks = [pruning]
        worker = train(model,executor,train_data,val_data,saved_root,
                       epoch=1,batch_size=16,patient=1,
                       use_gffcompare=False,
                       other_callbacks=other_callbacks,
                       add_grad=False,add_seq_fig=False)
        #Save best result ass final result
        trial.report(worker.best_result['val_loss'])
        if pruning.is_prune:
            raise TrialPruned()
        return worker.best_result['val_loss']
    return evaluate

def objective_generator(saved_root,data,spcae_generator,executor_config,**kwargs):
    evaluate = evaluate_generator(data,executor_config)
    def objective(trial):
        model_config = spcae_generator(trial,**kwargs)
        path = os.path.join(saved_root,"trial_{}").format(trial.number)
        result = evaluate(path,model_config,trial)
        return result
    return objective

def main(saved_root,objective_generator,spcae_generator,
         train_data_path,val_data_path,executor_config_path,**kwargs):
    setting = {'saved_root':saved_root,
               'train_data_path':train_data_path,
               'val_data_path':val_data_path,
               'executor_config_path':executor_config_path}
    #Create folder
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)
    copy_path(saved_root,executor_config_path)    
    
    #Save setting
    setting_path = os.path.join(saved_root,"main_setting.json")
    with open(setting_path,"w") as fp:
        json.dump(setting, fp, indent=4)

    #Load, parse and save data
    train_data = dd.io.load(train_data_path)
    val_data = dd.io.load(val_data_path)
    train_data = train_data[0],simplify_genome(train_data[1],SIMPLIFY_MAP)
    val_data = val_data[0],simplify_genome(val_data[1],SIMPLIFY_MAP)
    data = train_data, val_data
    write_fasta(os.path.join(saved_root,'train.fasta'),train_data[0])
    write_fasta(os.path.join(saved_root,'val.fasta'),val_data[0])    
    
    data_path = os.path.join(saved_root,"data.h5")
    if not os.path.exists(data_path):
        dd.io.save(data_path,data)

    #Create executor
    with open(executor_config_path,"r") as fp:
        executor_config = json.load(fp)

    objective = objective_generator(saved_root,data,spcae_generator,executor_config,**kwargs)
    study = optuna.create_study(study_name='seq_ann',
                                storage='sqlite:///{}/trials.db'.format(saved_root),
                                load_if_exists=True,
                                pruner=NopPruner(),
                                sampler=SkoptSampler())
    study.optimize(objective, n_trials=args.n_trials)

if __name__ == '__main__':
    main(args.saved_root,objective_generator,HAGRU_space_generator,
         args.train_data_path,args.val_data_path,
         args.executor_config_path,
         use_common_atten=args.use_common_atten)
