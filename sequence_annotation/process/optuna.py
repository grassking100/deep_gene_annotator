import torch
import os
import warnings
import enum
import datetime
import optuna
import pandas as pd
from abc import abstractmethod, ABCMeta, abstractproperty
from optuna.structs import FrozenTrial, TrialState
from optuna.trial import Trial
from optuna.pruners import NopPruner
from optuna.integration import SkoptSampler
from optuna.storages.rdb import models
from .callback import Callback, Callbacks
from ..utils.utils import read_json, write_json, create_folder
from ..utils.utils import from_time_str, replace_utc_to_local, to_time_str
from .director import check_max_memory_usgae


class OptunaCallback(Callback):
    def __init__(self, study, trial, target=None):
        self.trial = trial
        self.study = study
        self.target = target or 'val_loss'
        self._counter = None
        self._worker = None

    @property
    def has_updated(self):
        return self._counter is not None

    def on_work_begin(self, worker, **kwargs):
        self._counter = None
        self._worker = worker

    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['target'] = self.target
        return config

    def on_epoch_begin(self, counter, **kwargs):
        self._counter = counter

    def on_epoch_end(self, metric, **kwargs):
        target = metric[self.target]
        if str(target) == 'nan':
            return

        self.trial.report(target, self._counter)

def _config_to_json(config):
    if isinstance(config, dict):
        config_ = {}
        for key, value in config.items():
            config_[key] = _config_to_json(value)
        return config_
    elif isinstance(config, list):
        config_ = []
        for value in config:
            config_.append(_config_to_json(value))
        return config_
    elif isinstance(config, datetime.datetime):
        return to_time_str(replace_utc_to_local(config))
    elif isinstance(config, optuna.distributions.BaseDistribution):
        return str(config)
    elif isinstance(config, enum.Enum):
        return str(config)
    else:
        return config


def get_trial_config(trial):
    config = {}
    keys = [
        'number',
        'datetime_start',
        'params',
        'distributions',
        'user_attrs',
        'system_attrs']
    for key in keys:
        if hasattr(trial, key):
            config[key] = getattr(trial, key)
    return config


def get_frozen_trial_config(trial):
    config = {}
    keys = ['number', 'datetime_start', 'params', 'distributions', 'user_attrs', 'system_attrs',
            'value', 'datetime_complete', 'intermediate_values', 'state']
    for key in keys:
        if hasattr(trial, key):
            config[key] = getattr(trial, key)
    return config


def _config_to_frozen_trial(config):
    config = dict(config)
    config['datetime_start'] = from_time_str(
        config['datetime_start']).astimezone()
    if 'datetime_complete' in config:
        config['datetime_complete'] = from_time_str(
            config['datetime_complete']).astimezone()
    else:
        config['datetime_complete'] = None
    distributions = dict(config['distributions'])
    config['distributions'] = {}
    for key, value in distributions.items():
        config['distributions'][key] = eval(
            "optuna.distributions.{}".format(value))
    config["state"] = TrialState.COMPLETE
    if 'intermediate_values' not in config:
        config['intermediate_values'] = {}
    if 'value' not in config:
        config['value'] = None
    return FrozenTrial(**config, trial_id=config['number'] + 1)

def _shift_config_number(config, shift_number):
    config['number'] += shift_number
    config["system_attrs"]["_number"] += shift_number
    return config


def add_exist_trial(study, folder_path, trial_start_number=None):
    trial_start_number = trial_start_number or 0
    folder_name = folder_path.split('/')[-1]
    path = os.path.join(folder_path, '{}_result.json'.format(folder_name))
    if not os.path.exists(path):
        path = os.path.join(folder_path, '{}_config.json'.format(folder_name))
    print("Load result from {}".format(path))
    config = read_json(path)
    config = _shift_config_number(config, trial_start_number)
    trial = _config_to_frozen_trial(config)
    try:
        study._storage.create_new_trial(study.study_id, template_trial=trial)
    except BaseException:
        print("Fail to load trial {} from {}".format(trial.number, path))
        raise


def add_exist_trials(study, trial_list_path, trial_start_number=None):
    trial_start_number = trial_start_number or 0
    trials_root = '/'.join(trial_list_path.split('/')[:-1])
    df = pd.read_csv(trial_list_path,sep='\t',index_col=False,
                     dtype={'trial_number': int})
    df = df.drop_duplicates()
    df = df.sort_values(by=['trial_number'])['path']
    for path in list(df):
        print("Add trial from {}".format(path))
        trial_path=os.path.join(trials_root,path)
        add_exist_trial(study,trial_path,
                        trial_start_number)


def get_discrete_uniform(trial, key, lb=None, ub=None, step=None, value=None):
    step = step or 1
    if value is None:
        if lb is None or ub is None:
            raise Exception("Bound of {} must not be None".format(key))
        value = trial.suggest_discrete_uniform(key, lb, ub, step)
    return value


def get_choices(trial, name, choices, value=None):
    if value is None:
        value = trial.suggest_categorical(name, choices)
    return value


def get_bool_choices(trial, name, value=None):
    return get_choices(trial, name, [False, True], value)


def create_study(output_root, **kwargs):
    storage_path = 'sqlite:///{}/trials.db'.format(output_root)
    study = optuna.create_study(
        study_name='seq_ann',
        storage=storage_path,
        **kwargs)
    return study


class IModelExecutorCreator(metaclass=ABCMeta):
    """Creator to create model and executor by trial parameters"""
    @abstractmethod
    def create_default_hyperparameters(self):
        pass

    @abstractmethod
    def create_max(self):
        pass

    @abstractmethod
    def get_trial_config(self, trial, set_by_trial_config=False):
        pass

    @abstractmethod
    def create_all_configs(self):
        pass

    @abstractmethod
    def get_each_space_size(self):
        pass

    @abstractmethod
    def create_by_trial(self, trial, set_by_trial_config=False,
                        set_by_grid_search=False):
        pass

    @abstractproperty
    def space_size(self):
        pass


def _get_status(study, id_):
    session = study._storage.scoped_session()
    recorded_trial = models.TrialModel.find_or_raise_by_id(id_, session)
    print(recorded_trial.state)


class OptunaTrainer:
    def __init__(self, train_method, output_root, model_executor_creator,
                 train_data, val_data, epoch, batch_size, monitor_target,
                 is_minimize=True, by_grid_search=False, acq_func=None,
                 **train_kwargs):
        if is_minimize:
            self._direction = 'minimize'
        else:
            self._direction = 'maximize'
        self._save_distribution = False
        self._trial_start_number = 0
        self._monitor_target = monitor_target
        self._output_root = output_root
        self._train_method = train_method
        self._train_data = train_data
        self._val_data = val_data
        self._epoch = epoch
        self._batch_size = batch_size
        self._creator = model_executor_creator
        self._trial_list_path = os.path.join(self._output_root, "trial_list.tsv")
        self._train_kwargs = train_kwargs
        self._by_grid_search = by_grid_search
        self._study = None
        self._acq_func = acq_func or 'gp_hedge'

    def _get_trial_id(self, trial):
        return self._trial_start_number + trial.number

    def _get_trial_name(self, trial):
        return "trial_{}".format(self._get_trial_id(trial))

    def _record_trial_path(self, trial):
        name = self._get_trial_name(trial)
        exist = False
        if os.path.exists(self._trial_list_path):
            df = pd.read_csv(self._trial_list_path, sep='\t')
            if name in list(df['path']):
                exist = True

        if not exist:
            with open(self._trial_list_path, "a") as fp:
                fp.write("{}\t{}\n".format(self._get_trial_id(trial), name))

    def _get_trial_root(self, trial):
        return os.path.join(self._output_root, self._get_trial_name(trial))

    def _get_trial_config_path(self, trial):
        root = self._get_trial_root(trial)
        trial_name = self._get_trial_name(trial)
        return os.path.join(root, '{}_config.json'.format(trial_name))

    def _get_trial_result_path(self, trial):
        root = self._get_trial_root(trial)
        trial_name = self._get_trial_name(trial)
        return os.path.join(root, '{}_result.json'.format(trial_name))

    def _objective(self, trial, set_by_trial_config=False):
        # Set path
        trial_root = self._get_trial_root(trial)
        create_folder(trial_root)
        trial_config_path = self._get_trial_config_path(trial)
        # Check memory
        print("Check memory")
        temp = self._creator.create_by_trial(trial, set_by_trial_config=set_by_trial_config,
                                             set_by_grid_search=self._by_grid_search)
        temp_model = temp['model']
        temp_executor = temp['executor']
        check_max_memory_usgae(trial_root, temp_model, temp_executor,
                               self._train_data, self._val_data,
                               self._batch_size)
        del temp_model
        del temp_executor
        torch.cuda.empty_cache()
        print("Memory is available")
        # Start training
        created = self._creator.create_by_trial(trial, set_by_trial_config=set_by_trial_config,
                                                set_by_grid_search=self._by_grid_search)
        model = created['model']
        executor = created['executor']          
        model.save_distribution = self._save_distribution
        # save settings
        
        self._record_trial_path(trial)
        config = _config_to_json(get_trial_config(trial))
        config = _shift_config_number(config, self._trial_start_number)
        if not os.path.exists(trial_config_path):
            write_json(config, trial_config_path)
        # Train
        optuna_callback = OptunaCallback(
            self._study, trial, target=self._monitor_target)
        callbacks = Callbacks([optuna_callback])
        worker = self._train_method(trial_root, self._epoch, model, executor, self._train_data, self._val_data,
                                    batch_size=self._batch_size, other_callbacks=callbacks,
                                    **self._train_kwargs)

        # Save best result as final result
        best_result = worker.best_result[self._monitor_target]
        print(
            "Return {} as best result of trial {}".format(
                best_result,
                trial.number + 1))
        return best_result

    def _save_trial_result(self, frozen_trial):
        if not isinstance(frozen_trial, FrozenTrial):
            raise Exception("Result trail must be FrozenTrial")
        result_path = self._get_trial_result_path(frozen_trial)
        config = _config_to_json(get_frozen_trial_config(frozen_trial))
        _shift_config_number(config, self._trial_start_number)
        write_json(config, result_path)

    def _callback(self, study, frozen_trial):
        self._save_trial_result(frozen_trial)

    def _create_study(self, n_startup_trials=None):
        n_startup_trials = n_startup_trials or 0
        skopt_kwargs = {'n_initial_points': 0, 'acq_func':self._acq_func}
        warnings.warn(
            "The n_initial_points in skopt_kwargs is set to 0, PLEASE MAKE SURE the optuna version is >=1.2.0\n\n")
        sampler = SkoptSampler(
            skopt_kwargs=skopt_kwargs,
            n_startup_trials=n_startup_trials)
        study = create_study(self._output_root, direction=self._direction, load_if_exists=True,
                             pruner=NopPruner(), sampler=sampler)
        if not os.path.exists(self._trial_list_path):
            with open(self._trial_list_path, "w") as fp:
                fp.write("trial_number\tpath\n")
        return study

    def _get_resumed_trial(self, study, trial_number):
        trial_id = trial_number + 1 - self._trial_start_number
        session = study._storage.scoped_session()
        recorded_trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        recorded_trial.state = TrialState.RUNNING
        recorded_trial.value = 0
        study._storage._commit_with_integrity_check(session)
        print("Resume trial id {}".format(trial_id))
        trial = Trial(study, trial_id)
        record_path = os.path.join(
            self._get_trial_root(trial),
            'checkpoint/train_record.json')
        if os.path.exists(record_path):
            print(
                "Load existed record {} to trial {}".format(
                    record_path, trial_id))
            record = read_json(record_path)
            for index, value in enumerate(record[self._monitor_target]):
                trial.report(value, index + 1)
        trial.set_system_attr('fail_reason', float('nan'))
        return trial

    def train_single_trial(self, existed_trial_config,force=False):
        trial_number = existed_trial_config['number']
        trial_id = trial_number + 1
        self._study = self._create_study()
        trial = self._study.get_trials()[trial_number]
        if not force:
            if trial.state == TrialState.COMPLETE and trial.value is not None:
                print("Complete")
                return
        trial = self._get_resumed_trial(self._study, trial_number)
        try:
            result = self._objective(trial, set_by_trial_config=True)
        except:
            self._study._storage.set_trial_state(trial_id, TrialState.FAIL)
            raise

        session = self._study._storage.scoped_session()
        self._study._storage.set_trial_value(trial_id, result)
        self._study._storage.set_trial_state(trial_id, TrialState.COMPLETE)
        self._study._log_completed_trial(trial, result)
        self._study._storage._commit_with_integrity_check(session)
        frozen_trial = self._study.get_trials()[trial_number]
        self._save_trial_result(frozen_trial)

    def optimize(self, n_startup_trials=None, n_trials=None):
        n_startup_trials = n_startup_trials or 0
        n_trials = n_trials or self._creator.space_size
        create_folder(self._output_root)
        space_config_path = os.path.join(self._output_root, 'space_config.json')
        default_hyperparameters = self._creator.create_default_hyperparameters()
        if os.path.exists(space_config_path):
            existed = read_json(space_config_path)
            if default_hyperparameters != existed:
                raise Exception(
                    "The {} is not same as previous one".format(space_config_path))
        else:
            write_json(default_hyperparameters, space_config_path)

        print("Check max memory")
        max_result = self._creator.create_max()
        max_model = max_result['model']
        max_exec = max_result['executor']
        check_max_memory_usgae(self._output_root, max_model, max_exec,
                               self._train_data, self._val_data, self._batch_size)
        del max_model
        del max_exec
        torch.cuda.empty_cache()
        print("Memory is available")

        self._study = self._create_study(n_startup_trials)
        self._study.optimize(self._objective, n_trials=n_trials,
                             callbacks=[self._callback])
