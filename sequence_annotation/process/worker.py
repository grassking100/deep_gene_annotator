"""This submodule provides trainer to train model"""
import warnings
import os,sys
import time
import json
import torch
from abc import ABCMeta, abstractmethod
from .data_generator import SeqDataset,SeqGenerator
from .executor import BasicExecutor
from .callback import Accumulator, Recorder, Callbacks, WarningRecorder
from .warning import WorkerProtectedWarning
from ..utils.utils import create_folder

def _batch_process(model,seq_data,process,callbacks):
    callbacks.on_batch_begin()
    torch.cuda.empty_cache()
    ids,inputs,labels,lengths,seqs = seq_data.data
    inputs = inputs.cuda()
    labels = labels.cuda()
    returned = process(model,inputs,labels,lengths=lengths)
    metric,outputs,lengths,masks = returned
    seq_data = SeqDataset([ids,inputs,labels,lengths,seqs])
    callbacks.on_batch_end(outputs=outputs,
                           seq_data=seq_data,
                           metric=metric,
                           masks=masks)

def train_per_batch(model,seq_data,executor,callbacks):
    _batch_process(model,seq_data,executor.fit,callbacks)

def evaluate_per_batch(model,seq_data,executor,callbacks):
    _batch_process(model,seq_data,executor.evaluate,callbacks)
    
class Worker(metaclass=ABCMeta):
    def __init__(self,model,data,executor=None,path=None):
        super().__init__()
        self.model = model
        self.data = data
        self.executor = executor
        self.result = {}
        self.is_verbose_visible = True
        self._settings = {}
        self.path = path
        if executor is None:
            self.executor = BasicExecutor()
        if self.path is not None:
            create_folder(self.path)

    def work(self):
        """Work"""
        self._before_work()
        self._work()
        self._after_work()
            
    @abstractmethod
    def _work(self):
        """Work"""
        pass

    @abstractmethod
    def _after_work(self):
        """Do something after worker work"""
        pass

    @abstractmethod
    def _before_work(self):
        """Do something before worker work"""
        pass
 
    def print_verbose(self,info):
        if self.is_verbose_visible:
            print(info,end='\r')
            sys.stdout.write('\033[K')

class TrainWorker(Worker):
    """a worker which will train and evaluate the model"""
    def __init__(self,model,data,executor=None,train_generator=None,val_generator=None,
                 train_callbacks=None,val_callbacks=None,other_callbacks=None,
                 writer=None,epoch_start=None,epoch=None,path=None):
        super().__init__(model,data,executor,path=path)
        self._train_generator = train_generator
        self._val_generator = val_generator
        self._train_callbacks = train_callbacks
        self._val_callbacks = val_callbacks
        self._other_callbacks = other_callbacks

        if train_generator is None:
            self._train_generator = SeqGenerator()
        if val_generator is None:
            self._val_generator = SeqGenerator(shuffle=False)
        if train_callbacks is None:
            self._train_callbacks = Callbacks()
        if val_callbacks is None:
            self._val_callbacks = Callbacks()
        if other_callbacks is None:
            self._other_callbacks = Callbacks()

        self._writer = writer
        self._epoch_start = epoch_start or 0
        self._epoch = 1 if epoch is None else epoch
        self._best_epoch = None
        self._is_running = True
        self._recoder = None
        self._best_result = None

    @property
    def best_result(self):
        return self._best_result
    
    @property
    def is_running(self):
        return self._is_running
    
    @is_running.setter
    def is_running(self,value):
        warnings.warn("Is_running SHOULD only be called by callback", WorkerProtectedWarning)
        self._is_running = value
        
    @property
    def best_epoch(self):
        return self._best_epoch
    
    @best_epoch.setter
    def best_epoch(self,value):
        warnings.warn("Best_epoch SHOULD only be called by callback", WorkerProtectedWarning)
        self._best_epoch = value

    def _save_setting(self):
        if self.path is not None:
            for key in ['_train_callbacks','_val_callbacks','_other_callbacks',
                        '_epoch_start','_epoch','executor']:
                attr = getattr(self,key)
                if hasattr(attr,'get_config'):
                    attr = attr.get_config()
                self._settings[key] = attr

            with open(os.path.join(self.path,"train_worker_setting.json"),'w') as fp:
                json.dump(self._settings,fp, indent=4)

    def _before_work(self):
        self._train_loader = self._train_generator(SeqDataset(self.data['training']))
        if 'validation' in self.data.keys():
            self._val_loader = self._val_generator(SeqDataset(self.data['validation']))
            acc = Accumulator(prefix='val')
            self._val_callbacks.add(acc)
        record_path = None
        warning_path = None
        if self.path is not None:    
            record_path = os.path.join(self.path,"train_record.json")
            warning_path = os.path.join(self.path,"warning.txt")    
        accumulator = Accumulator()
        warning_recorder = WarningRecorder(path=warning_path)
        self._train_callbacks.add(accumulator)
        self._other_callbacks.add(warning_recorder)
        self._recoder = Recorder(path=record_path)
        self.executor.process(self.model)
        self._is_running = True
        self._save_setting()

    def _work(self):
        """Train model"""
        all_callbacks = Callbacks([self._train_callbacks,self._val_callbacks,
                                   self._other_callbacks,self._recoder])
        all_callbacks.on_work_begin(worker=self)
        start = self._epoch_start+1
        end = start+self._epoch
        batch_info = "Epoch: ({}/{}), {} {:.1f}% of data"
        epoch_info = "Epoch: ({}/{}), Time cost of: {}, {}"
        for epoch in range(start,end):
            pre_time = time.time()
            if self._writer is not None:
                self._writer.counter = epoch
            all_callbacks.on_epoch_begin(counter=epoch)
            warnings = []
            for index,item in enumerate(self._train_loader):
                seq_data = SeqDataset(item)
                train_per_batch(self.model,seq_data,self.executor,self._train_callbacks)
                status = 100*index/len(self._train_loader)
                self.print_verbose(batch_info.format(epoch,self._epoch,'training',status))
                for name,param in self.model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.cpu().detach().numpy()
                        if  (grad>1).any():
                            warning = "Epoch {}: {} have gradients which are larger than one, the max value is {}".format(epoch,name,grad.max())
                            print(warning)
                            warnings.append(warning)

            for index,item in enumerate(self._val_loader):
                seq_data = SeqDataset(item)
                evaluate_per_batch(self.model,seq_data,self.executor,self._val_callbacks)
                status = 100*index/len(self._val_loader)
                self.print_verbose(batch_info.format(epoch,self._epoch,'validating',status))

            train_record = self._train_callbacks.get_data()
            val_record = self._val_callbacks.get_data()
            other_record = self._other_callbacks.get_data()
            if str(train_record['loss']) == 'nan':
                self._is_running = False
            if 'val_loss' in val_record.keys() and str(val_record['val_loss']) == 'nan':
                self._is_running = False

            record = {}
            record.update(train_record)
            record.update(val_record)
            record.update(other_record)
            self._train_callbacks.on_epoch_end(metric=train_record)
            self._val_callbacks.on_epoch_end(metric=val_record)
            self._other_callbacks.on_epoch_end(metric=record,warnings=warnings)
            self._recoder.on_epoch_end(metric=record)

            time_cost = round(time.time()-pre_time,3)
            self.print_verbose(epoch_info.format(epoch,self._epoch,time_cost,record))

            if not self.is_running:
                self.print_verbose("Early stop at {}".format(epoch))
                break

        all_callbacks.on_work_end()

    def _after_work(self):
        self.result = self._recoder.data
        if self.best_epoch is not None:
            self._best_result = {}
            for key,value in self.result.items():
                self._best_result[key] = self.result[key][self.best_epoch - 1]
            if self.path is not None:    
                best_path = os.path.join(self.path,"best_record.json")
                best_result = {'best_epoch':self.best_epoch,
                               'best_result':self.best_result}
                with open(best_path,'w') as fp:
                    json.dump(best_result,fp)

class TestWorker(Worker):
    """a worker which will evaluate the model"""
    def __init__(self,model,data,executor=None,generator=None,callbacks=None,path=None):
        super().__init__(model,data,executor,path=path)
        self._generator = generator
        self._callbacks = callbacks
        self._recoder = None

        if generator is None:
            self._generator = DataGenerator(shuffle=False)
        if callbacks is None:
            self._callbacks = Callbacks()

    def _save_setting(self):
        if self.path is not None:
            for key in ['_callbacks']:
                attr = getattr(self,key)
                if hasattr(attr,'get_config'):
                    attr = attr.get_config()
                self._settings[key] = attr
            with open(os.path.join(self.path,"test_worker_setting.json"),'w') as fp:
                json.dump(self._settings,fp, indent=4)
            
    def _before_work(self):
        self._loader = self._generator(SeqDataset(self.data['testing']))
        
        record_path = None
        if self.path is not None:   
            record_path = os.path.join(self.path,"test_record.json")
        accum = Accumulator(prefix='test')
        self._recoder = Recorder(path=record_path)
        self._callbacks.add(accum)
        self.executor.process(self.model)
        self._save_setting()

    def _work(self):
        """Test model"""
        callbacks = Callbacks(self._callbacks)
        self._recoder.on_work_begin(worker=self)
        callbacks.on_work_begin(worker=self)
        record = {}
        self._recoder.on_epoch_begin()
        callbacks.on_epoch_begin(counter=1)
        batch_info = "Testing data: {0:.1f}%"

        for index,item in enumerate(self._loader):
            seq_data = SeqDataset(item)
            evaluate_per_batch(self.model,seq_data,
                               self.executor,self._callbacks)
            status = 100*index/len(self._loader)
            self.print_verbose(batch_info.format(status))

        record = callbacks.get_data()
        self._recoder.on_epoch_end(metric=record)
        callbacks.on_epoch_end(metric=record)
        callbacks.on_work_end()
        self._recoder.on_work_end()

    def _after_work(self):
        self.result = self._recoder.data
