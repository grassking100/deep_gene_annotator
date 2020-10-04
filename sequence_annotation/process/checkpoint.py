import abc
import os
import json
import torch
from ..utils.utils import write_json, create_folder, read_json, get_time_str
from .utils import get_copied_state_dict
from .callback import Callback, Callbacks,get_prefix

def copy_file(source_path, target_path):
    command = 'cp {} {}'.format(source_path, target_path)
    if os.path.exists(source_path):
        os.system(command)

def get_best_result(best_epoch, result):
    best_result = {}
    for key, value in result.items():
        if len(result[key]) < best_epoch:
            prompt = "The length of data, {}, at best epoch, {}, cause conflict"
            raise Exception(prompt.format(len(result[key]), best_epoch))
        best_result[key] = result[key][best_epoch - 1]
    return best_result

def _get_name(path):
    return path.split('/')[-1].split('.')[0]

def _read_status(root, read_path):
    with open(read_path, "r") as fp:
        status = json.load(fp)
    epoch = int(status['epoch'])
    if 'relative_path' in status:
        path = os.path.join(root, status['relative_path'])
    else:
        path = status['path']
    return epoch, path

def _write_status(saved_path, epoch, weights_path):
    relative_path = weights_path.split('/')[-1]
    status = {"epoch": epoch,"relative_path": relative_path,"path": weights_path,
              "record_time": get_time_str()}
    write_json(status, saved_path)

def _write_best_status(saved_path, best_epoch, best_target, weights_path, patience):
    relative_path = weights_path.split('/')[-1]
    best_status = {'best_epoch': best_epoch,'relative_path': relative_path,
                   "path": weights_path,'best_target': best_target,
                   'patience': patience,"record_time": get_time_str(),
                   'formula': '(self._counter-self.best_epoch) >= self._patience'}
    write_json(best_status, saved_path)

def _save_best(model_path, status_path, best_epoch, best_target, model_weights, patience):
    print("Save best model of epoch {} at {}".format(best_epoch, model_path))
    torch.save(model_weights, model_path)
    _write_best_status(status_path, best_epoch, best_target, model_path,patience)

class Recorder(Callback):
    def __init__(self, prefix=None, path=None, force_reset=False):
        self._prefix = get_prefix(prefix)
        self._path = path
        self._force_reset = force_reset
        self._start_epoch = None
        self._epoch = None
        self._data = {}

    @property
    def path(self):
        return self._path
        
    @property
    def epoch(self):
        return self._epoch
        
    @property
    def data(self):
        return self._data

    def set_start_epoch(self, value):
        self._start_epoch = value

    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['prefix'] = self._prefix
        config['path'] = self._path
        config['force_reset'] = self._force_reset
        return config

    def _reset(self):
        self._data = {'record_time': [], 'epoch': []}
        if self._path is not None and os.path.exists(self._path) and not self._force_reset:
            print("Load record from {}".format(self._path))
            with open(self._path, 'r') as fp:
                self._data.update(json.load(fp))

    def on_work_begin(self, **kwargs):
        self._reset()
        if self._start_epoch is not None:
            for type_, value in self._data.items():
                self._data[type_] = value[:self._start_epoch]

    def on_epoch_begin(self, counter, **kwargs):
        self._epoch = counter

    def on_epoch_end(self, metric, **kwargs):
        for type_, value in metric.items():
            if type_ not in self._data.keys():
                self._data[type_] = []
            self._data[type_].append(value)
        if self._path is not None:
            data = dict(self._data)
            data['record_time'].append(get_time_str())
            data['epoch'].append(self._epoch)
            write_json(data, self._path)
    
class _Checkpoint(Callback):
    def __init__(self, root):
        self._root = root
        self._counter = None
        self._worker = None
        self._has_updated = False

    @property
    def counter(self):
        return self._counter
        
    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['root'] = self._root
        return config

    @abc.abstractproperty
    def paths(self):
        pass


class ModelCheckpoint(_Checkpoint):
    def __init__(self,model, root,target=None,optimize_min=True,
                 patience=None,warmup_epoch=None):
        super().__init__(root)
        self._warmup_epoch = warmup_epoch or 0
        self._target = target or 'val_loss'
        self._optimize_min = optimize_min
        self._patience = patience
        self._counter = None
        self._best_result = None
        self._best_epoch = None
        self._model = model
        self._best_weights = None
        self.latest_status_path = os.path.join(self._root,'latest_model.status')
        self.best_status_path = os.path.join(self._root, 'best_model.status')
        self.latest_model_path = os.path.join(self._root, 'latest_model.pth')
        self.best_model_path = os.path.join(root, 'best_model.pth')

    @property
    def target(self):
        return self._target
        
    @property
    def paths(self):
        paths = {}
        paths['best'] = {'status': self.best_status_path,'weights': self.best_model_path}
        paths['latest'] = {'status': self.latest_status_path,'weights': self.latest_model_path}
        return paths

    def should_stop(self):
        if self._patience is not None:
            return (self._counter - self.best_epoch) >= self._patience
        else:
            return False

    def _stop_worker(self):
        stop_prompt = "{} stop worker because {}-{}>={}"
        print(stop_prompt.format(self.__class__.__name__, self._counter,
                                 self.best_epoch,self._patience))
        self._worker.set_running_status(False)

    def on_work_begin(self, worker, **kwargs):
        self._has_updated = False
        self._worker = worker
        self._best_result = None
        self._best_epoch = 0
        self._best_weights = None
        #self._model = self._worker.executor.model
        # Load model weights
        biggest_epoch = 0
        model_path = None
        if os.path.exists(self.latest_status_path):
            biggest_epoch, model_path = _read_status(self._root,self.latest_status_path)
        if model_path is not None:
            print("Load model of epoch {} from {}".format(biggest_epoch, model_path))
            self._model.load(model_path)
        self._counter = biggest_epoch

        # Load best model weights, epoch, and value
        if os.path.exists(self.best_status_path):
            best_status = read_json(self.best_status_path)
            best_epoch = int(best_status['best_epoch'])
            best_target = float(best_status['best_target'])
            self._best_weights = torch.load(self.best_model_path,map_location='cpu')
            self._best_epoch = best_epoch
            self._best_result = best_target
            print("Load existed best model of epoch {} from {}".format(best_epoch, self.best_model_path))
            print("Load best target, {}, of epoch {} from {}".format(best_target, best_epoch, self.best_status_path))
            if self._counter < self._best_epoch:
                inconsist_prompt = "Best epoch {} should not exceed current epoch {}"
                raise Exception(inconsist_prompt.format(self._best_epoch, self._counter))

        if self.should_stop():
            self._stop_worker()

    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['warmup_epoch'] = self._warmup_epoch
        config['target'] = self._target
        config['optimize_min'] = self._optimize_min
        config['patience'] = self._patience
        return config

    @property
    def best_result(self):
        return self._best_result

    @property
    def best_epoch(self):
        return self._best_epoch

    def on_epoch_begin(self, counter, **kwargs):
        self._counter = counter
        if self._counter > self._warmup_epoch:
            self._has_updated = True
            
    def _save_model(self):
        #Record model
        model_path = self.latest_model_path
        print("Save model of epoch {} at {}".format(self._counter, model_path))
        self._model.save(model_path,True)
        _write_status(self.latest_status_path, self._counter, model_path)
        
    def _save_best_model(self,target):
        update = False
        if self._best_result is None:
            update = True
        else:
            if self._optimize_min:
                if self._best_result > target:
                    update = True
            else:
                if self._best_result < target:
                    update = True

        if update:
            print("Save best weight of epoch {}".format(self._counter))
            self._best_weights = get_copied_state_dict(self._model)
            self._best_epoch = self._counter
            self._best_result = target
            print("Update best weight of epoch {}".format(self._counter))
            _save_best(self.best_model_path, self.best_status_path,
                       self.best_epoch, self._best_result,
                       self._best_weights, self._patience)

            
    def on_epoch_end(self, metric, **kwargs):
        target = metric[self._target]
        self._save_model()
        if self._counter > self._warmup_epoch:    
            self._save_best_model(target)
        if self.should_stop():
            self._stop_worker()

    def on_work_end(self):
        print("Best " + str(self._target) + ": " + str(self._best_result))
        print("Restore best weight of epoch {}".format(self._best_epoch))
        self._model.load_state_dict(self._best_weights)
        _save_best(self.best_model_path, self.best_status_path,
                   self.best_epoch, self._best_result,
                   self._best_weights, self._patience)


class ExecutorCheckpoint(_Checkpoint):
    def __init__(self, executor, root):
        super().__init__(root)
        self._executor = executor
        self.latest_status_path = os.path.join(self._root,'latest_executor.status')
        self.latest_executor_path = os.path.join(self._root,'latest_executor.pth')

    @property
    def paths(self):
        paths = {}
        paths['latest'] = {'status': self.latest_status_path,'weights': self.latest_executor_path}
        return paths

    def on_work_begin(self, worker, **kwargs):
        self._has_updated = False
        #self._executor = worker.executor
        executor_path = None
        biggest_epoch = 0
        if os.path.exists(self.latest_status_path):
            biggest_epoch, executor_path = _read_status(self._root, self.latest_status_path)

        if executor_path is not None:
            print("Load executor of epoch {} from {}".format(biggest_epoch, executor_path))
            self._executor.load(executor_path)
        self._counter = biggest_epoch

    def on_epoch_begin(self, counter, **kwargs):
        self._counter = counter
        self._has_updated = True

    def on_epoch_end(self, metric, **kwargs):
        executor_path = self.latest_executor_path
        print("Save executor of epoch {} at {}".format(self._counter, executor_path))
        self._executor.save(executor_path,True)
        _write_status(self.latest_status_path, self._counter, executor_path)

class Checkpoint(Callback):
    def __init__(self, model_checkpoint, executor_checkpoint, recorder,
                 checkpoint_root, best_record_path):
        self.model_checkpoint = model_checkpoint
        self.executor_checkpoint = executor_checkpoint
        self.recorder = recorder
        self.checkpoint_root = checkpoint_root
        self.callbacks = Callbacks([self.model_checkpoint,
                                    self.executor_checkpoint,
                                    self.recorder])
        self._start_epoch = 0
        self._best_record_path = best_record_path
        self._best_result = None
        self._best_epoch = None
        self._record = None
        self._has_updated = False

    @property
    def record(self):
        return self._record

    @property
    def best_result(self):
        return self._best_result

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_record_path(self):
        return self._best_record_path

    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['best_record_path'] = self._best_record_path
        config['checkpoint_root'] = self.checkpoint_root
        config['model_checkpoint'] = self.model_checkpoint.get_config(**kwargs)
        config['executor_checkpoint'] = self.executor_checkpoint.get_config(**kwargs)
        config['recorder'] = self.recorder.get_config(**kwargs)
        return config

    def _save_checkpoint(self):
        if not self._has_updated:
            return
        name = _get_name(self.recorder.path)
        epoch = self.recorder.epoch
        new_record_path = '{}_epoch_{}.json'.format(name, epoch)
        new_record_path = os.path.join(self.checkpoint_root, new_record_path)
        if not os.path.exists(new_record_path):
            copy_file(self.recorder.path, new_record_path)

        for path_dict in [self.executor_checkpoint.paths, self.model_checkpoint.paths]:
            for dict_ in path_dict.values():
                patience = None
                status_path = dict_['status']
                weights_path = dict_['weights']
                if os.path.exists(status_path):
                    status = read_json(status_path)
                    is_best_related = 'best_epoch' in status
                    if is_best_related:
                        epoch = status['best_epoch']
                        best_target = status['best_target']
                        if 'patience' in status:
                            patience = status['patience']
                    else:
                        epoch = status['epoch']

                    new_status_name = _get_name(status_path)
                    new_weights_name = _get_name(weights_path)
                    new_status_path = '{}_epoch_{}.status'.format(new_status_name, epoch)
                    new_weights_path = '{}_epoch_{}.pth'.format(new_weights_name, epoch)
                    new_status_path = os.path.join(self.checkpoint_root,new_status_path)
                    new_weights_path = os.path.join(self.checkpoint_root,new_weights_path)

                    if not os.path.exists(new_status_path):
                        if is_best_related:
                            _write_best_status(new_status_path, epoch,best_target, 
                                               new_weights_path,patience)
                        else:
                            _write_status(new_status_path, epoch,
                                          new_weights_path)
                    if not os.path.exists(new_weights_path):
                        copy_file(weights_path, new_weights_path)

    def on_work_begin(self, worker, **kwargs):
        self._has_updated = False
        self._worker = worker
        self._best_result = None
        self._best_epoch = 0
        self._model_weights = None
        create_folder(self.checkpoint_root)
        with open(os.path.join(self.checkpoint_root,'status.txt'),'w') as fp:
            fp.write("")

        if os.path.exists(self.model_checkpoint.best_status_path) and os.path.exists(self.best_record_path):
            best_record = read_json(self.best_record_path)
            best_status = read_json(self.model_checkpoint.best_status_path)
            # Validate best epoch
            best_record_epoch = best_record['best_epoch']
            best_status_epoch = int(best_status['best_epoch'])
            if best_record_epoch != best_status_epoch:
                inconsist_prompt = "Inconsist best epoch between {} and {}"
                raise Exception(inconsist_prompt.format(best_record_epoch, best_status_epoch))

        self.model_checkpoint.on_work_begin(worker=worker, **kwargs)
        self.executor_checkpoint.on_work_begin(worker=worker, **kwargs)
        if self.model_checkpoint.counter != self.executor_checkpoint.counter:
            message = "Checkpoint at model and executor are not the same"
            ", got {} and {}".format(self.model_checkpoint.counter,
                                     self.executor_checkpoint.counter)
            raise Exception(message)
        self._start_epoch = self.model_checkpoint.counter
        self.recorder.set_start_epoch(self._start_epoch)
        self.recorder.on_work_begin(worker=worker, **kwargs)
        self._save_checkpoint()
        worker.set_start_epoch(self._start_epoch + 1)

    def on_work_end(self, **kwargs):
        self.callbacks.on_work_end(**kwargs)
        self._save_checkpoint()
        self._record = self.recorder.data
        inconsist_prompt = "Inconsist best result between {} and {} at epoch {}"
        if self.model_checkpoint.best_epoch is not None:
            best_result = get_best_result(self.model_checkpoint.best_epoch,self.record)
            best_target_at_result = best_result[self.model_checkpoint.target]
            if best_target_at_result != self.model_checkpoint.best_result:
                raise Exception(inconsist_prompt.format(best_target_at_result,
                                                        self.model_checkpoint.best_result,
                                                        self.model_checkpoint.best_epoch))
            else:
                with open(os.path.join(self.checkpoint_root,'status.txt'),'w') as fp:
                    fp.write("Finished")
            self._best_result = best_result
            self._best_epoch = self.model_checkpoint.best_epoch
            if self._has_updated and self.best_record_path is not None:
                print("Save best result of epoch {}".format(self.model_checkpoint.best_epoch))
                best_result = get_best_result(self.model_checkpoint.best_epoch,self.record)
                best_result = {'best_epoch': self.model_checkpoint.best_epoch,
                               'best_result': best_result,'record_time': get_time_str()}
                write_json(best_result, self.best_record_path)

    def on_epoch_begin(self, **kwargs):
        self._has_updated = True
        self.callbacks.on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs):
        self.callbacks.on_epoch_end(**kwargs)

    def on_batch_begin(self,**kwargs):
        self.callbacks.on_batch_begin(**kwargs)

    def on_batch_end(self, **kwargs):
        self.callbacks.on_batch_end(**kwargs)


def build_checkpoint(root,prefix,model=None,executor=None,force_reset=False,
                     monitor_target=None,patience=None,warmup_epoch=None):
    record_path = os.path.join(root, "{}_record.json".format(prefix))
    recorder = Recorder(path=record_path, force_reset=force_reset)
    if model is None or executor is None:
        return recorder
    else:
        best_record_path = os.path.join(root, 'best_record.json')
        model_checkpoint = ModelCheckpoint(model,target=monitor_target,patience=patience,
                                           root=root,warmup_epoch=warmup_epoch)
        executor_checkpoint = ExecutorCheckpoint(executor,root=root)
        checkpoint = Checkpoint(model_checkpoint, executor_checkpoint,
                                recorder, root, best_record_path)

        return checkpoint
