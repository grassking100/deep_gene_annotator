import os
import json
import torch
import warnings
from ..utils.utils import write_json
from .warning import WorkerProtectedWarning
from .utils import get_copied_state_dict
from .callback import Callback,Callbacks

def _read_status(root,read_path):
    with open(read_path,"r") as fp:
        status = json.load(fp)
    epoch = int(status['epoch'])
    if 'relative_path' in status:
        path = os.path.join(root,status['relative_path'])
    else:
        path = status['path']
    return epoch,path
    
def _read_best_status(root,read_path):
    with open(read_path,"r") as fp:
        status = json.load(fp)
    epoch = int(status['best_epoch'])
    best_target = float(status['best_target'])
    if 'relative_path' in status:
        path = os.path.join(root,status['relative_path'])
    else:
        path = status['path']
    return epoch,best_target,path
          
def _write_status(root,saved_path,epoch,relative_path):
    path = os.path.join(root,relative_path)
    status = {"epoch":epoch,"relative_path":relative_path,"path":path}
    write_json(status,saved_path)
        
def save_best(best_model_path,best_status_path,best_epoch,best_target,model_weights):
    relative_path = best_model_path.split('/')[-1]
    print("Save best model of epoch {} at {}".format(best_epoch,best_model_path))
    torch.save(model_weights,best_model_path)
    best_status = {'best_epoch':best_epoch,
                   'relative_path':relative_path,
                   "path":best_model_path,
                   'best_target':best_target,
                   'formula':'(self._counter-self.best_epoch) >= self.patient'}
    write_json(best_status,best_status_path)

class Checkpoint(Callback):
    def __init__(self,path,prefix=None,period=None):
        super().__init__(prefix)
        if path is None:
            raise Exception("Path should not be None")
        self.path = path
        self._counter = None
        self._worker = None
        self.period = period or 1
        
    @property
    def counter(self):
        return self._counter
    
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self.path
        config['period'] = self.period
        return config

class ModelCheckpoint(Checkpoint):
    def __init__(self,path,target=None,optimize_min=True,
                 patient=None,period=None,save_best_weights=False,
                 restore_best_weights=False,prefix=None,
                 explicit_period=False):
        super().__init__(path,prefix=prefix,period=period)
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
        self.last_status_path = os.path.join(self.path,'last_model.status')
        self.latest_status_path = os.path.join(self.path,'latest_model.status')
        self.best_status_path = os.path.join(self.path,'best_model.status')
        self.last_model_path =  os.path.join(self.path,'last_model.pth')
        self.latest_model_path =  os.path.join(self.path,'latest_model.pth')
        self.best_model_path = os.path.join(path,'best_model.pth')
        
    def on_work_begin(self, worker,**kwargs):
        self._worker = worker
        self._best_result = None
        self._best_epoch = 0
        self._model_weights = None

        #Load best model weights, epoch, and value   
        if self.save_best_weights:
            if os.path.exists(self.best_status_path):
                best_epoch,best_result,best_model_path = _read_best_status(self.path,self.best_status_path)
                self._model_weights = torch.load(best_model_path)
                self._best_epoch = best_epoch
                self._best_result = best_result
                print("Load existed best model of epoch {} from {}".format(best_epoch,best_model_path))
                print("Load best record, {}, of epoch {} from {}".format(best_result,best_epoch,best_model_path))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore",category=WorkerProtectedWarning)
                    self._worker.best_epoch = self._best_epoch
            else:
                self._model_weights = get_copied_state_dict(worker.model)

        #Load model weights
        biggest_epoch = 0
        model_path = None
        if os.path.exists(self.latest_status_path):
            biggest_epoch,model_path = _read_status(self.path,self.latest_status_path)
        if os.path.exists(self.last_status_path):
            biggest_epoch_,model_path_ = _read_status(self.path,self.last_status_path)
            if biggest_epoch_ >= biggest_epoch:
                biggest_epoch = biggest_epoch_
                model_path = model_path_
        if model_path is not None:
            print("Load epoch {}'s model from {}".format(biggest_epoch,model_path))
            self._worker.model.load_state_dict(torch.load(model_path),strict=True)
        self._counter = biggest_epoch

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=WorkerProtectedWarning)
                self._worker.best_epoch = self._best_epoch
            if self.save_best_weights:
                save_best(self.best_model_path,self.best_status_path,
                          self.best_epoch,self._best_result,self._model_weights)
        
        if self.patient is not None:
            if (self._counter-self.best_epoch) >= self.patient:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore",category=WorkerProtectedWarning)
                    self._worker.is_running = False

        if (self._counter%self.period) == 0:
            if self.explicit_period:
                relative_path = 'model_epoch_{}.pth'.format(self._counter)
            else:
                relative_path = 'latest_model.pth'
            model_path = os.path.join(self.path,relative_path)
            print("Save model at "+model_path)
            torch.save(self._worker.model.state_dict(),model_path)
            _write_status(self.path,self.latest_status_path,
                          self._counter,relative_path)

    def on_work_end(self):
        torch.save(self._worker.model.state_dict(),self.last_model_path)
        print("Save last model of epoch {} at {}".format(self._counter,self.last_model_path))
        _write_status(self.path,self.last_status_path,
                      self._counter,'last_model.pth')

        print("Best "+str(self.target)+": "+str(self._best_result))
        if self.save_best_weights:
            if self.restore_best_weights:
                print("Restore best weight of epoch {}".format(self._best_epoch))
                self._worker.model.load_state_dict(self._model_weights)
            save_best(self.best_model_path,self.best_status_path,
                      self.best_epoch,self._best_result,self._model_weights)
            
class ExecutorCheckpoint(Checkpoint):
    def __init__(self,path,period=None,prefix=None,explicit_period=False):
        super().__init__(path,prefix=prefix,period=period)
        self._executor_state_dict = None
        self._executor = None
        self.explicit_period = explicit_period
        #Path
        self.last_status_path = os.path.join(self.path,'last_executor.status')
        self.latest_status_path = os.path.join(self.path,'latest_executor.status')
        self.latest_executor_path =  os.path.join(self.path,'latest_executor.pth')
        
    def on_work_begin(self, worker,**kwargs):
        self._executor = worker.executor
        self._executor_state_dict = self._executor.state_dict()

        executor_path = None
        biggest_epoch = 0
        if os.path.exists(self.latest_status_path):
            biggest_epoch,executor_path = _read_status(self.path,self.latest_status_path)
                
        if os.path.exists(self.last_status_path):
            biggest_epoch_,executor_path_ = _read_status(self.path,self.last_status_path)
            if biggest_epoch_ >= biggest_epoch:
                biggest_epoch = biggest_epoch_
                executor_path = executor_path_
            
        if executor_path is not None:
            print("Load epoch {}'s executor from {}".format(biggest_epoch,executor_path))
            self._executor.load_state_dict(torch.load(executor_path))
        self._counter = biggest_epoch

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['explicit_period'] = self.explicit_period
        return config

    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter

    def on_epoch_end(self,metric,**kwargs):
        self._executor_state_dict = self._executor.state_dict()
        if (self._counter%self.period) == 0:
            if self.explicit_period:
                relative_path = 'executor_epoch_{}.pth'.format(self._counter)
            else:
                relative_path = 'latest_executor.pth'
            executor_path = os.path.join(self.path,relative_path)
            print("Save epoch {}'s executor at {}".format(self._counter,executor_path))
            torch.save(self._executor_state_dict,executor_path)
            _write_status(self.path,self.latest_status_path,
                          self._counter,relative_path)

    def on_work_end(self):
        relative_path = 'last_executor.pth'
        executor_path =  os.path.join(self.path,relative_path)
        torch.save(self._executor_state_dict,executor_path)
        print("Save last epoch {}'s of executor at {}".format(self._counter,executor_path))
        _write_status(self.path,self.last_status_path,
                      self._counter,relative_path)

class ModelExecutorCheckpoint(Callback):
    def __init__(self,model_checkpoint,executor_checkpoint,prefix=None):
        super().__init__(prefix)
        self.model_checkpoint = model_checkpoint
        self.executor_checkpoint = executor_checkpoint
        self.callbacks = Callbacks([self.model_checkpoint,self.executor_checkpoint])
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['model_checkpoint'] = self.model_checkpoint.get_config(**kwargs)
        config['executor_checkpoint'] = self.executor_checkpoint.get_config(**kwargs)
        return config

    def on_work_begin(self,**kwargs):
        self.callbacks.on_work_begin(**kwargs)
        if self.model_checkpoint.counter != self.executor_checkpoint.counter:
            message = "Checkpoint at model and executor are not the same"
            ", got {} and {}".format(self.model_checkpoint.counter,
                                     self.executor_checkpoint.counter)
            raise Exception(message)

    def on_work_end(self):
        self.callbacks.on_work_end()

    def on_epoch_begin(self,**kwargs):
        self.callbacks.on_epoch_begin(**kwargs)

    def on_epoch_end(self,**kwargs):
        self.callbacks.on_epoch_end(**kwargs)

    def on_batch_begin(self):
        self.callbacks.on_batch_begin(**kwargs)

    def on_batch_end(self,**kwargs):
        self.callbacks.on_batch_end(**kwargs)
        
def use_checkpoint(path,monitor_target=None,patient=None,period=None):
    model_checkpoint = ModelCheckpoint(target=monitor_target,patient=patient,
                                       save_best_weights=True,restore_best_weights=True,
                                       path=path,period=period)
    executor_checkpoint = ExecutorCheckpoint(path=path,period=period)
    checkpoint = ModelExecutorCheckpoint(model_checkpoint,executor_checkpoint)
    
    #If best record exist, then write best target to best_model_status.json
    best_record_path = os.path.join(path,'best_record.json')
    if model_checkpoint.save_best_weights and os.path.exists(best_record_path):
        with open(model_checkpoint.best_status_path,"r") as fp:
            best_status = json.load(fp)
        if 'best_target' not in best_status:                    
            with open(best_record_path,"r") as fp:
                best_record = json.load(fp)
            best_epoch = best_record['best_epoch']
            best_target = best_record['best_result'][model_checkpoint.target]
            if best_epoch != best_status['best_epoch']:
                raise Exception()
            best_status['best_target'] = best_target
            print("Write best target, {}, to {}".format(best_target,model_checkpoint.best_status_path))
            write_json(best_status,model_checkpoint.best_status_path)
        
    epoch_start = 0
    if os.path.exists(model_checkpoint.latest_status_path):
        with open(model_checkpoint.latest_status_path,"r") as fp:
            latest_status = json.load(fp)
            epoch_start = max(epoch_start,latest_status['epoch'])

    if os.path.exists(model_checkpoint.last_status_path):
        with open(model_checkpoint.last_status_path,"r") as fp:
            last_status = json.load(fp)
            epoch_start = max(epoch_start,last_status['epoch'])
    return epoch_start,checkpoint