import os
import json
import torch
import warnings
from ..utils.utils import write_json,create_folder,read_json,get_time_str
from .warning import WorkerProtectedWarning
from .utils import get_copied_state_dict,deep_copy
from .callback import Callback, Callbacks, DataCallback
from .warning import CheckpointProtectedWarning

class Recorder(DataCallback):
    def __init__(self,prefix=None,path=None,force_reset=False):
        super().__init__(prefix)
        self.path = path
        self._force_reset = force_reset
        self._epoch_start = None
        self._epoch = None

    @property
    def epoch(self):
        return self._epoch
        
    @property
    def epoch_start(self):
        return self._epoch_start
        
    @epoch_start.setter
    def epoch_start(self,value):
        warnings.warn("epoch_start SHOULD only be called by checkpoint", CheckpointProtectedWarning)
        self._epoch_start = value
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self.path
        config['force_reset'] = self._force_reset
        return config

    def _reset(self):
        self._data = {'record_time':[],'epoch':[]}
        if self.path is not None and os.path.exists(self.path) and not self._force_reset:
            print("Load record from {}".format(self.path))
            with open(self.path,'r') as fp:
                self._data.update(json.load(fp))

    def on_work_begin(self,**kwargs):
        super().on_work_begin()
        if self.epoch_start is not None:
            for type_,value in self._data.items():
                self._data[type_] = value[:self.epoch_start]

    def on_epoch_begin(self,counter,**kwargs):
        self._epoch = counter
                
    def on_epoch_end(self,metric,**kwargs):
        for type_,value in metric.items():
            if type_ not in self._data.keys():
                self._data[type_] = []
            self._data[type_].append(value)

        if self.path is not None:
            data = dict(self._data)
            data['record_time'].append(get_time_str())
            data['epoch'].append(self._epoch)
            write_json(data,self.path)

    @property
    def data(self):
        return self._data

def _read_status(root,read_path):
    with open(read_path,"r") as fp:
        status = json.load(fp)
    epoch = int(status['epoch'])
    if 'relative_path' in status:
        path = os.path.join(root,status['relative_path'])
    else:
        path = status['path']
    return epoch,path
      
def _write_status(saved_path,epoch,weights_path):
    relative_path = weights_path.split('/')[-1]
    status = {"epoch":epoch,"relative_path":relative_path,"path":weights_path,"record_time":get_time_str()}
    write_json(status,saved_path)
    
def _write_best_status(saved_path,best_epoch,best_target,weights_path,patient):
    relative_path = weights_path.split('/')[-1]
    best_status = {'best_epoch':best_epoch,
                   'relative_path':relative_path,
                   "path":weights_path,
                   'best_target':best_target,
                   'patient':patient,
                   'formula':'(self._counter-self.best_epoch) >= self.patient',
                   "record_time":get_time_str()}
    write_json(best_status,saved_path)
        
def _save_best(model_path,status_path,best_epoch,best_target,model_weights,patient):
    print("Save best model of epoch {} at {}".format(best_epoch,model_path))
    torch.save(model_weights,model_path)
    _write_best_status(status_path,best_epoch,best_target,model_path,patient)
    
class _Checkpoint(Callback):
    def __init__(self,root,period=None):
        if root is None:
            raise Exception("Root should not be None")
        self.root = root
        self._counter = None
        self._worker = None
        self.period = period or 1
        self._has_updated = False
        
    @property
    def counter(self):
        return self._counter
    
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['root'] = self.root
        config['period'] = self.period
        return config
    
    @property
    def paths(self):
        pass

class ModelCheckpoint(_Checkpoint):
    def __init__(self,root,target=None,optimize_min=True,
                 patient=None,period=None,save_best_weights=False,
                 restore_best_weights=False,
                 explicit_period=False):
        super().__init__(root,period=period)
        self.target = target or 'val_loss'
        self.optimize_min = optimize_min
        self.patient = patient
        self.save_best_weights = save_best_weights
        self.restore_best_weights = restore_best_weights
        self._counter = None
        self._best_result = None
        self._best_epoch = None
        self._model_weights = None
        self.explicit_period = explicit_period
        #Path
        self.last_status_path = os.path.join(self.root,'last_model.status')
        self.latest_status_path = os.path.join(self.root,'latest_model.status')
        self.best_status_path = os.path.join(self.root,'best_model.status')
        self.last_model_path =  os.path.join(self.root,'last_model.pth')
        self.latest_model_path =  os.path.join(self.root,'latest_model.pth')
        self.best_model_path = os.path.join(root,'best_model.pth')
        
    @property
    def paths(self):
        paths = {}
        paths['last'] = {'status':self.last_status_path,
                         'weights':self.last_model_path}
        
        paths['best'] = {'status':self.best_status_path,
                         'weights':self.best_model_path}
        
        paths['latest'] = {'status':self.latest_status_path,
                         'weights':self.latest_model_path}
        
        return paths

    def should_stop(self):
        if self.patient is not None:
            return (self._counter-self.best_epoch) >= self.patient
        else:
            return False

    def _stop_worker(self):
        print("{} stop worker because {}-{}>={}".format(self.__class__.__name__,
                                                        self._counter,
                                                        self.best_epoch,
                                                        self.patient))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=WorkerProtectedWarning)
            self._worker.is_running = False
        
    def on_work_begin(self, worker,**kwargs):
        self._has_updated = False
        self._worker = worker
        self._best_result = None
        self._best_epoch = 0
        self._model_weights = None
        #Load model weights
        biggest_epoch = 0
        model_path = None
        if os.path.exists(self.latest_status_path):
            biggest_epoch,model_path = _read_status(self.root,self.latest_status_path)
        if os.path.exists(self.last_status_path):
            biggest_epoch_,model_path_ = _read_status(self.root,self.last_status_path)
            if biggest_epoch_ >= biggest_epoch:
                biggest_epoch = biggest_epoch_
                model_path = model_path_
        if model_path is not None:
            print("Load model of epoch {} from {}".format(biggest_epoch,model_path))
            self._worker.model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=True)
        self._counter = biggest_epoch
        
        #Load best model weights, epoch, and value   
        if self.save_best_weights:
            if os.path.exists(self.best_status_path):
                best_status = read_json(self.best_status_path)
                best_epoch = int(best_status['best_epoch'])
                best_target =  float(best_status['best_target'])
                self._model_weights = torch.load(self.best_model_path, map_location='cpu')
                self._best_epoch = best_epoch
                self._best_result = best_target
                print("Load existed best model of epoch {} from {}".format(best_epoch,self.best_model_path))
                print("Load best target, {}, of epoch {} from {}".format(best_target,best_epoch,self.best_status_path))
                if self._counter < self._best_result:
                    raise Exception("Best epoch {} should not exceed current epoch {}".format(self._best_result,self._counter))
            else:
                self._model_weights = get_copied_state_dict(worker.model)
        
        if self.should_stop():
            self._stop_worker()

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['target'] = self.target
        config['optimize_min'] = self.optimize_min
        config['patient'] = self.patient
        config['save_best_weights'] = self.save_best_weights
        config['restore_best_weights'] = self.restore_best_weights
        config['explicit_period'] = self.explicit_period
        return config

    @property
    def best_result(self):
        return self._best_result
    @property
    def best_epoch(self):
        return self._best_epoch
    
    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter
        self._has_updated = True

    def on_epoch_end(self,metric,**kwargs):
        target = metric[self.target]
        if str(target) == 'nan':
            return
        update = False
        if self._best_result is None:
            update = True
        else:
            if self.optimize_min:
                if self._best_result > target:
                    update = True
            else:
                if self._best_result < target:
                    update = True

        if update:
            if self.save_best_weights:
                print("Save best weight of epoch {}".format(self._counter))
                self._model_weights = get_copied_state_dict(self._worker.model)
            self._best_epoch = self._counter
            self._best_result = target
            print("Update best weight of epoch {}".format(self._counter))

            if self.save_best_weights:
                _save_best(self.best_model_path,self.best_status_path,
                           self.best_epoch,self._best_result,
                           self._model_weights,self.patient)
        
        if self.should_stop():
            self._stop_worker()

        if (self._counter%self.period) == 0:
            if self.explicit_period:
                relative_path = 'model_epoch_{}.pth'.format(self._counter)
                model_path = os.path.join(self.root,relative_path)
            else:
                model_path = self.latest_model_path

            print("Save model of epoch {} at {}".format(self._counter,model_path))
            torch.save(self._worker.model.state_dict(),model_path)
            _write_status(self.latest_status_path,self._counter,model_path)

    def on_work_end(self):
        if self._has_updated:
            torch.save(self._worker.model.state_dict(),self.last_model_path)
            print("Save last model of epoch {} at {}".format(self._counter,self.last_model_path))
            _write_status(self.last_status_path,self._counter,self.last_model_path)

            print("Best "+str(self.target)+": "+str(self._best_result))
            if self.save_best_weights:
                if self.restore_best_weights:
                    print("Restore best weight of epoch {}".format(self._best_epoch))
                    self._worker.model.load_state_dict(self._model_weights)
                _save_best(self.best_model_path,self.best_status_path,
                           self.best_epoch,self._best_result,
                           self._model_weights,self.patient)
            
class ExecutorCheckpoint(_Checkpoint):
    def __init__(self,root,period=None,explicit_period=False):
        super().__init__(root,period=period)
        self._executor_state_dict = None
        self._executor = None
        self.explicit_period = explicit_period
        #Path
        self.last_status_path = os.path.join(self.root,'last_executor.status')
        self.latest_status_path = os.path.join(self.root,'latest_executor.status')
        self.latest_executor_path = os.path.join(self.root,'latest_executor.pth')
        self.last_executor_path =  os.path.join(self.root,'last_executor.pth')
        
    @property
    def paths(self):
        paths = {}
        paths['last'] = {'status':self.last_status_path,
                         'weights':self.last_executor_path}

        paths['latest'] = {'status':self.latest_status_path,
                         'weights':self.latest_executor_path}
        
        return paths
        
    def on_work_begin(self, worker,**kwargs):
        self._has_updated = False
        self._executor = worker.executor
        self._executor_state_dict = deep_copy(self._executor.state_dict())

        executor_path = None
        biggest_epoch = 0
        if os.path.exists(self.latest_status_path):
            biggest_epoch,executor_path = _read_status(self.root,self.latest_status_path)
                
        if os.path.exists(self.last_status_path):
            biggest_epoch_,executor_path_ = _read_status(self.root,self.last_status_path)
            if biggest_epoch_ >= biggest_epoch:
                biggest_epoch = biggest_epoch_
                executor_path = executor_path_
            
        if executor_path is not None:
            print("Load executor of epoch {} from {}".format(biggest_epoch,executor_path))
            self._executor.load_state_dict(torch.load(executor_path, map_location='cpu'))
        self._counter = biggest_epoch

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['explicit_period'] = self.explicit_period
        return config

    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter
        self._has_updated = True

    def on_epoch_end(self,metric,**kwargs):
        self._executor_state_dict = deep_copy(self._executor.state_dict())
        if (self._counter%self.period) == 0:
            if self.explicit_period:
                relative_path = 'executor_epoch_{}.pth'.format(self._counter)
                executor_path = os.path.join(self.root,relative_path)
            else:
                executor_path = self.latest_executor_path

            print("Save executor of epoch {} at {}".format(self._counter,executor_path))
            torch.save(self._executor_state_dict,executor_path)
            _write_status(self.latest_status_path,self._counter,executor_path)

    def on_work_end(self):
        if self._has_updated:
            torch.save(self._executor_state_dict,self.last_executor_path)
            print("Save executor of last epoch {} at {}".format(self._counter,self.last_executor_path))
            _write_status(self.last_status_path,self._counter,self.last_executor_path)

def copy_file(source_path,target_path):
    command = 'cp {} {}'.format(source_path,target_path)
    if os.path.exists(source_path):
        os.system(command)
        
def get_best_result(best_epoch,result):
    best_result = {}
    for key,value in result.items():
        if len(result[key]) < best_epoch:
            raise Exception("The length of data, {}, at best epoch, {}, cause conflict".format(len(result[key]),best_epoch))
        best_result[key] = result[key][best_epoch-1]
    return best_result

def _get_name(path):
    return path.split('/')[-1].split('.')[0]

class Checkpoint(Callback):
    def __init__(self,model_checkpoint,executor_checkpoint,recorder,checkpoint_root,best_record_path):
        self.model_checkpoint = model_checkpoint
        self.executor_checkpoint = executor_checkpoint
        self.recorder = recorder
        self.checkpoint_root = checkpoint_root
        self.callbacks = Callbacks([self.model_checkpoint,self.executor_checkpoint,self.recorder])
        self._epoch_start = 0
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
    def weights_status_paths(self):
        paths = [self.executor_checkpoint.paths,
                 self.model_checkpoint.paths]
        return paths
      
    @property
    def best_record_path(self):
        return self._best_record_path

    def get_config(self,**kwargs):
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
        new_record_path = '{}_epoch_{}.json'.format(name,epoch)
        new_record_path = os.path.join(self.checkpoint_root,new_record_path)
        if not os.path.exists(new_record_path):
            copy_file(self.recorder.path,new_record_path)
        
        for path_dict in self.weights_status_paths:
            for dict_ in path_dict.values():
                patient = None
                status_path = dict_['status']
                weights_path = dict_['weights']
                if os.path.exists(status_path):
                    status = read_json(status_path)
                    is_best_related='best_epoch' in status
                    if is_best_related:
                        epoch = status['best_epoch']
                        best_target = status['best_target']
                        if 'patient' in status:
                            patient = status['patient']
                    else:
                        epoch = status['epoch']

                    new_status_name = _get_name(status_path)
                    new_weights_name = _get_name(weights_path)
                    new_status_path = '{}_epoch_{}.status'.format(new_status_name,epoch)
                    new_weights_path = '{}_epoch_{}.pth'.format(new_weights_name,epoch)
                    new_status_path = os.path.join(self.checkpoint_root,new_status_path)
                    new_weights_path = os.path.join(self.checkpoint_root,new_weights_path)

                    if not os.path.exists(new_status_path):
                        if is_best_related:
                            _write_best_status(new_status_path,epoch,best_target,new_weights_path,patient)
                        else:
                            _write_status(new_status_path,epoch,new_weights_path)
                    if not os.path.exists(new_weights_path):
                        copy_file(weights_path,new_weights_path)
         
    @property
    def epoch_start(self):
        return self._epoch_start
        
    def on_work_begin(self, worker,**kwargs):
        create_folder(self.checkpoint_root)
        self._has_updated = False
        self._worker = worker
        self._best_result = None
        self._best_epoch = 0
        self._model_weights = None
 
        if self.model_checkpoint.save_best_weights:
            if os.path.exists(self.model_checkpoint.best_status_path) and os.path.exists(self.best_record_path):
                best_record = read_json(self.best_record_path)
                best_status = read_json(self.model_checkpoint.best_status_path)
                #Validate best epoch
                best_record_epoch = best_record['best_epoch']
                best_status_epoch = int(best_status['best_epoch'])
                if best_record_epoch != best_status_epoch:
                    raise Exception("Inconsist best epoch between {} and {}".format(best_record_epoch,best_status_epoch))

        self.model_checkpoint.on_work_begin(worker=worker,**kwargs)
        self.executor_checkpoint.on_work_begin(worker=worker,**kwargs)
        if self.model_checkpoint.counter != self.executor_checkpoint.counter:
            message = "Checkpoint at model and executor are not the same"
            ", got {} and {}".format(self.model_checkpoint.counter,
                                     self.executor_checkpoint.counter)
            raise Exception(message)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=CheckpointProtectedWarning)
            self._epoch_start = self.model_checkpoint.counter
            self.recorder.epoch_start = self._epoch_start
        self.recorder.on_work_begin(worker=worker,**kwargs)
        self._save_checkpoint()

    def on_work_end(self,**kwargs):
        self.callbacks.on_work_end(**kwargs)
        self._save_checkpoint()
        self._record = self.recorder.data
        if self.model_checkpoint.best_epoch is not None:
            best_result = get_best_result(self.model_checkpoint.best_epoch,self.record)
            best_target_at_result = best_result[self.model_checkpoint.target]
            if best_target_at_result != self.model_checkpoint.best_result:
                raise Exception("Inconsist best result between {} and {} at epoch {}".format(best_target_at_result,
                                                                                             self.model_checkpoint.best_result,
                                                                                             self.model_checkpoint.best_epoch))
            self._best_result = best_result
            self._best_epoch = self.model_checkpoint.best_epoch
            if self._has_updated and self.best_record_path is not None:
                print("Save best result of epoch {}".format(self.model_checkpoint.best_epoch))
                best_result = get_best_result(self.model_checkpoint.best_epoch,self.record)
                best_result = {'best_epoch':self.model_checkpoint.best_epoch,
                               'best_result':best_result,
                               'record_time':get_time_str()
                              }
                write_json(best_result,self.best_record_path)

    def on_epoch_begin(self,**kwargs):
        self._has_updated = True
        self.callbacks.on_epoch_begin(**kwargs)

    def on_epoch_end(self,**kwargs):
        self.callbacks.on_epoch_end(**kwargs)

    def on_batch_begin(self):
        self.callbacks.on_batch_begin(**kwargs)

    def on_batch_end(self,**kwargs):
        self.callbacks.on_batch_end(**kwargs)

def build_checkpoint(root,only_recorder=False,force_reset=None,
                     monitor_target=None,patient=None,period=None,is_train_mode=False):
    if is_train_mode:
        record_path = os.path.join(root,"train_record.json")
    else:
        record_path = os.path.join(root,"test_record.json")

    recorder = Recorder(path=record_path,force_reset=force_reset)
    if only_recorder:
        return recorder
    else:
        best_record_path = os.path.join(root,'best_record.json')
        model_checkpoint = ModelCheckpoint(target=monitor_target,patient=patient,
                                           save_best_weights=True,restore_best_weights=True,
                                           root=root,period=period)
        executor_checkpoint = ExecutorCheckpoint(root=root,period=period)

        checkpoint = Checkpoint(model_checkpoint,executor_checkpoint,recorder,root,best_record_path)
    
        return checkpoint
