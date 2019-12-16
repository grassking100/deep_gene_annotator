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
    parser.add_argument("--n_warmup_steps",type=int,default=10)
    parser.add_argument("--interval_steps",type=int,default=10)
    parser.add_argument("--n_initial_points",type=int,default=10)
    parser.add_argument("--rnn_hidden_coef_max",type=int,default=None)
    parser.add_argument("--saved_trials_path")
    parser.add_argument("--not_use_first_atten",action='store_true')
    parser.add_argument("--not_use_second_atten",action='store_true')
    parser.add_argument("--rnn_type")
    parser.add_argument("--frozen_trial_path")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id

import sys
import datetime
from optuna.structs import TrialState
import json
import pickle
import deepdish as dd
import optuna
from optuna.pruners import MedianPruner,NopPruner
from optuna.structs import TrialPruned,FrozenTrial
from optuna.trial import Trial
from optuna.integration import SkoptSampler
from optuna import trial as trial_module
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import write_fasta, create_folder
from sequence_annotation.process.optuna import add_exist_trials
from sequence_annotation.process.callback import OptunaCallback
from sequence_annotation.genome_handler.ann_genome_processor import simplify_genome
from main.utils import load_data, get_model, get_executor, copy_path, SIMPLIFY_MAP
from main.train_model import train
from main.space_generator import Builder
    
def evaluate_generator(data,executor_config,epoch=None,batch_size=None):
    def evaluate(saved_root,model_config,trial,frozen_trial=None):
        if not os.path.exists(saved_root):
            os.mkdir(saved_root)
        if hasattr(trial,'set_user_attr'):
            trial.set_user_attr('resource_path',saved_root)
            trial.set_user_attr('generate_by_exist',False)
        path = os.path.join(saved_root,'trial.pkl') 
        trial_id = trial.trial_id
        params = trial.params
        distributions = trial.distributions
        if frozen_trial is not None:
            trial_id = frozen_trial.trial_id
            params = frozen_trial.params
            distributions = frozen_trial.distributions
        trial_number = trial_id - 1
        
        frozen = FrozenTrial(number=trial_number,trial_id=trial_id,
                             datetime_start=trial.datetime_start,params=params,
                             distributions=distributions,user_attrs=trial.user_attrs,
                             system_attrs={'_number':trial_number},                            
                             state=TrialState.RUNNING,
                             value=None,
                             datetime_complete=None,
                             intermediate_values={})

        with open(path,'wb') as fp:
            pickle.dump(frozen, fp)
        path = os.path.join(saved_root,'trial_{}_model_config.json'.format(trial_number))
        with open(path,"w") as fp:
            json.dump(model_config,fp, indent=4)
        path = os.path.join(saved_root,'trial_{}_executor_config.json'.format(trial_number))
        with open(path,"w") as fp:
            json.dump(executor_config,fp, indent=4)
        path = os.path.join(saved_root,'trial_{}_params.json'.format(trial_number))
        with open(path,"w") as fp:
            json.dump(params,fp, indent=4)

        train_data, val_data = data
        model = get_model(model_config)
        executor = get_executor(model,**executor_config)
        optuna_callback = OptunaCallback(trial)
        other_callbacks = [optuna_callback]
        worker = train(saved_root,epoch,model,executor,train_data,val_data,
                       batch_size=batch_size,other_callbacks=other_callbacks)
        #Save best result as final result
        trial.report(worker.best_result['val_loss'])
        
        path = os.path.join(saved_root,'trial.pkl') 
        if optuna_callback.is_prune:
            state = TrialState.PRUNED
        else:
            state = TrialState.COMPLETE
        
        range_ = list(range(1,1+len(worker.result['val_loss'])))
        frozen = FrozenTrial(number=trial_number,trial_id=trial_id,
                             datetime_start=trial.datetime_start,params=params,
                             distributions=distributions,user_attrs=trial.user_attrs,
                             system_attrs={'_number':trial_number},
                             state=state,
                             value=worker.best_result['val_loss'],
                             intermediate_values=dict(zip(range_,worker.result['val_loss'])),
                             datetime_complete=datetime.datetime.now())
        with open(path,'wb') as fp:
            pickle.dump(frozen, fp)

        if optuna_callback.is_prune:
            raise TrialPruned()
        return worker.best_result['val_loss']
    return evaluate

def objective_generator(saved_root,data,space_generator,executor_config,epoch=None,batch_size=None):
    evaluate = evaluate_generator(data,executor_config,epoch=epoch,batch_size=batch_size)
    trial_list_path = os.path.join(saved_root,"trial_list.tsv")
    if not os.path.exists(trial_list_path):
        with open(trial_list_path,"w") as fp:
            fp.write("trial_id\tpath\n")
    def objective(trial):
        model_config = space_generator(trial)
        path = os.path.join(saved_root,"trial_{}").format(trial.number)
        with open(trial_list_path,"a") as fp:
            fp.write("{}\t{}\n".format(trial.number,path))
        result = evaluate(path,model_config,trial)
        return result
    return objective

def fixed_trial_evaluate(study,saved_root,data,space_generator,executor_config,epoch=None,batch_size=None):
    evaluate = evaluate_generator(data,executor_config,epoch=epoch,batch_size=batch_size)
    with open(os.path.join(saved_root,'trial.pkl'),'rb') as fp:
        frozen_trial=pickle.load(fp)
    model_config = space_generator(frozen_trial)
    trial_id = study._storage.create_new_trial(0)
    trial = Trial(study,trial_id)
    result = evaluate(saved_root,model_config,trial,frozen_trial)
    return result

if __name__ == '__main__':
    builder = Builder()
    cnn_num = None
    if not args.use_cnn:
        cnn_num = 0
    
    from_trial_params=False
    if args.frozen_trial_path is not None:
        from_trial_params=True
    space_generator = builder.build(args.out_channels,args.last_act,
                                    cnn_num=cnn_num,rnn_type=args.rnn_type,
                                    rnn_hidden_coef_max=args.rnn_hidden_coef_max,
                                    use_first_atten= not args.not_use_first_atten,
                                    use_second_atten= not args.not_use_second_atten,
                                    from_trial_params=from_trial_params)

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
        print("Use pruner")
        pruner = MedianPruner(n_startup_trials=args.n_startup_trials,
                              n_warmup_steps=args.n_warmup_steps,
                              interval_steps=args.interval_steps)
    else:
        pruner = NopPruner()
    trials_path = os.path.join(args.saved_root,'trials')
    if not os.path.exists(trials_path):
        os.mkdir(trials_path)
    objective = objective_generator(trials_path,data,space_generator,executor_config,
                                    epoch=args.epoch,batch_size=args.batch_size)
    if args.frozen_trial_path is not None:
        study = optuna.create_study()
    else:
        skopt_kwargs={'n_initial_points':args.n_initial_points}
        storage_path = 'sqlite:///{}/trials.db'.format(trials_path)
        study = optuna.create_study(study_name='seq_ann',
                                    storage=storage_path,
                                    load_if_exists=True,
                                    pruner=pruner,
                                    sampler=SkoptSampler(skopt_kwargs=skopt_kwargs))
    if args.saved_trials_path is not None:
        print("Load exist trials from {} into study".format(args.saved_trials_path))
        add_exist_trials(study,args.saved_trials_path)
    if args.frozen_trial_path is not None:
        fixed_trial_evaluate(study,args.frozen_trial_path,data,space_generator,executor_config,
                             epoch=args.epoch,batch_size=args.batch_size)
    else:
        if len(study.trials) < args.n_trials:
            study.optimize(objective, n_trials=args.n_trials)
    