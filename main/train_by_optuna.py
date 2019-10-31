import os
import sys
import json
import deepdish as dd
from argparse import ArgumentParser
import optuna
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
    
def HAGRU_space_generator(trial,padding_handle=None):
    builder = SeqAnnBuilder()
    builder.feature_block_config['num_layers'] = trial.suggest_int('cnn_num_coef',1,8)*2
    builder.feature_block_config['cnn_setting']['out_channels'] = trial.suggest_int('cnn_out_coef',1,16)*2
    builder.feature_block_config['cnn_setting']['kernel_size'] = trial.suggest_int('cnn_kernel_coef',1,16)*8+1
    builder.feature_block_config['cnn_setting']['padding_handle'] = padding_handle or 'partial'
    builder.feature_block_config['cnn_setting']['customized_init_mode'] = 'xavier_uniform_zero_cnn_init'
    
    builder.relation_block_config['rnn_type'] = 'HierAttenGRU'
    builder.relation_block_config['rnn_setting']['num_layers'] = trial.suggest_int('rnn_num',1,4)
    builder.relation_block_config['rnn_setting']['hidden_size'] = trial.suggest_int('rnn_hidden_coef',1,16)*2
    builder.relation_block_config['rnn_setting']['customized_gru_init_mode'] = 'orth_xav_zero_gru_init'
    builder.relation_block_config['rnn_setting']['customized_cnn_init_mode'] = 'xavier_uniform_zero_cnn_init'
    
    
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
                       epoch=100,batch_size=4,patient=10,
                       use_gffcompare=False,
                       other_callbacks=other_callbacks,
                       add_grad=False,add_seq_fig=False)
        #Save best result ass final result
        trial.report(worker.best_result['val_loss'])
        if pruning.is_prune:
            raise TrialPruned()
        return worker.best_result['val_loss']
    return evaluate

def objective_generator(saved_root,data,spcae_generator,executor_config):
    evaluate = evaluate_generator(data,executor_config)
    def objective(trial):
        model_config = spcae_generator(trial)
        path = os.path.join(saved_root,"trial_{}").format(trial.number)
        result = evaluate(path,model_config,trial)
        return result
    return objective

def main(saved_root,objective_generator,spcae_generator,
         train_data_path,val_data_path,executor_config_path):
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

    objective = objective_generator(saved_root,data,spcae_generator,executor_config)
    study = optuna.create_study(study_name='seq_ann',
                              storage='sqlite:///{}/trials.db'.format(saved_root),
                              load_if_exists=True,
                              pruner=NopPruner(),
                              sampler=TPESampler())
    study.optimize(objective, n_trials=20)

if __name__ == '__main__':
    main(args.saved_root,objective_generator,HAGRU_space_generator,
         args.train_data_path,args.val_data_path,
         args.executor_config_path)
