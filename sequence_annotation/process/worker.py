"""This submodule provides trainer to train model"""
import signal
import warnings
import os
import time
import torch
from abc import ABCMeta, abstractmethod
from ..utils.utils import create_folder,print_progress,write_json,get_time_str
from .data_generator import SeqDataset,SeqGenerator
from .executor import BasicExecutor
from .callback import MeanRecorder, Callbacks,DataHolder
from .warning import WorkerProtectedWarning
from .checkpoint import build_checkpoint

def _batch_process(model,seq_data,process,callbacks,**kwargs):
    torch.cuda.empty_cache()
    callbacks.on_batch_begin()
    ids,inputs,labels,lengths,seqs,strands = seq_data.data
    inputs = inputs.cuda()
    labels = labels.cuda()
    returned = process(model,inputs,labels,lengths=lengths,**kwargs)
    torch.cuda.empty_cache()
    metric,predict_result,lengths,masks,outputs = returned
    seq_data = SeqDataset([ids,inputs,labels,lengths,seqs,strands])
    with torch.no_grad():
        callbacks.on_batch_end(predicts=predict_result,
                               outputs=outputs,seq_data=seq_data,
                               metric=metric,masks=masks)
    torch.cuda.empty_cache()

def train_per_batch(model,seq_data,executor,callbacks):
    _batch_process(model,seq_data,executor.fit,callbacks)

def evaluate_per_batch(model,seq_data,executor,callbacks):
    _batch_process(model,seq_data,executor.evaluate,callbacks,accumulate=True)

def validate_gradient(model,message_recorder=None):
    for name,param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.cpu().detach().numpy()
            if (grad>1).any():
                warning = gradient_warning.format(epoch,name,grad.max())
                print(warning)
                if message_recorder is not None:
                    message_recorder.notify([warning])
    
class MessageRecorder:
    def __init__(self,path=None):
        self.path = path
    
    def notify(self,messages):
        if not isinstance(messages,list):
            messages = [messages]
        if self.path is not None:
            with open(self.path,'a+') as fp:
                for message in messages:
                    fp.write("{}\n".format(message))

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
        self._message_recorder = None
        signal.signal(signal.SIGTERM, self.handle_signal)

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
 
    def print_verbose(self,info,is_progress=False):
        if self.is_verbose_visible:
            if is_progress:
                print_progress(info)
            else:
                print(info)
            
    def handle_signal(self, signum, frame):
        pass
    
class TrainWorker(Worker):
    """a worker which will train and evaluate the model"""
    def __init__(self,model,data,executor=None,train_generator=None,val_generator=None,
                 train_callbacks=None,val_callbacks=None,other_callbacks=None,
                 writer=None,epoch=None,path=None,checkpoint_kwargs=None):
        super().__init__(model,data,executor,path=path)
        self._train_generator = train_generator
        self._val_generator = val_generator
        self._train_callbacks = train_callbacks
        self._val_callbacks = val_callbacks
        self._other_callbacks = other_callbacks
        self.checkpoint_kwargs = checkpoint_kwargs or {}
        
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
        self._epoch = 1 if epoch is None else epoch
        self._current_epoch = None
        self._best_epoch = None
        self._best_result = None
        self._is_running = True
        self._checkpoint = None
        self._train_loader = None
        self._val_loader = None

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
   
    def _save_setting(self):
        if self.path is not None:
            for key in ['_train_callbacks','_val_callbacks','_other_callbacks','path',
                        '_epoch','executor','checkpoint_kwargs']:
                attr = getattr(self,key)
                if hasattr(attr,'get_config'):
                    attr = attr.get_config()
                self._settings[key] = attr
            path = os.path.join(self.path,"train_worker_setting.json")
            write_json(self._settings,path)

    def _before_work(self):
        self._train_loader = self._train_generator(SeqDataset(self.data['training']))
        self._val_loader = self._val_generator(SeqDataset(self.data['validation']))
        record_path = None
        message_path = None
        if self.path is not None:    
            record_path = os.path.join(self.path,"train_record.json")
            message_path = os.path.join(self.path,"message.txt")
            self._checkpoint = build_checkpoint(self.path,record_path=record_path,
                                                **self.checkpoint_kwargs)
        self._message_recorder = MessageRecorder(path=message_path)
        self._train_callbacks.add(MeanRecorder())
        self._val_callbacks.add(DataHolder(prefix='val'))
        self._is_running = True
        self._save_setting()

    def handle_signal(self, signum, frame):
        self._is_running = False
        warning = "STOP worker at epoch {} by signal {}".format(self._current_epoch,signum)
        self.print_verbose(warning)
        if self._message_recorder is not None:
            self._message_recorder.notify([warning])
            
    def work(self):
        #Execute worker
        pre_time = time.time()
        try:
            super().work()
        except RuntimeError:
            self._message_recorder.notify(["Something wrong happened"])
            raise
        time_spend = time.time() - pre_time
        time_messgae = "Time spend: {} seconds\n".format(time_spend)
        if self._message_recorder is not None:
            self._message_recorder.notify([time_messgae])
            
    def _work(self):
        """Train model"""
        start = 1
        end = self._epoch + 1
        all_callbacks = Callbacks([self._train_callbacks,self._val_callbacks,
                                   self._other_callbacks])
        if self._checkpoint is not None:
            all_callbacks.add(self._checkpoint)
            
        all_callbacks.on_work_begin(worker=self)
        
        if self._checkpoint is not None:
            start = self._checkpoint.epoch_start+1

        batch_info = "Epoch: ({}/{}), {} {:.1f}% of data"
        epoch_info = "Epoch: ({}/{}), Time cost of: {}, {}"
        gradient_warning = "Epoch {}: {} have gradients which are larger than one, the max value is {}"
        self.print_verbose("Start from {} to {}".format(start,end-1))
        if not self.is_running:
            self.print_verbose("Stop at {}".format(start))
        else:
            time_messgae = "Start trainging at {}".format(get_time_str())
            self.print_verbose(time_messgae)
            if self._message_recorder is not None:
                self._message_recorder.notify(time_messgae)
            save_distribution = self.model.save_distribution
            for epoch in range(start,end):
                self._current_epoch = epoch
                pre_time = time.time()
                if self._writer is not None:
                    self._writer.counter = epoch
                all_callbacks.on_epoch_begin(counter=epoch)
                self.model.save_distribution=False
                for index,item in enumerate(self._train_loader):
                    seq_data = SeqDataset(item)
                    train_per_batch(self.model,seq_data,self.executor,self._train_callbacks)
                    status = 100*index/len(self._train_loader)
                    self.print_verbose(batch_info.format(epoch,self._epoch,'training',status),True)
                    validate_gradient(self.model,message_recorder=self._message_recorder)

                if self._val_loader is not None:
                    for index,item in enumerate(self._val_loader):
                        seq_data = SeqDataset(item)
                        evaluate_per_batch(self.model,seq_data,self.executor,self._val_callbacks)
                        status = 100*index/len(self._val_loader)
                        self.print_verbose(batch_info.format(epoch,self._epoch,'validating',status),True)

                self.model.save_distribution = save_distribution
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

                self.executor.on_epoch_end(epoch=epoch,metric=record)
                self._train_callbacks.on_epoch_end(metric=train_record)
                self._val_callbacks.on_epoch_end(metric=val_record)
                self._other_callbacks.on_epoch_end(metric=record)
                if self._checkpoint is not None:
                    self._checkpoint.on_epoch_end(metric=record)

                time_cost = round(time.time()-pre_time,3)
                self.print_verbose(epoch_info.format(epoch,self._epoch,time_cost,record),True)

                if not self.is_running:
                    self.print_verbose("Stop at {}".format(epoch))                    
                    break
            time_messgae = "Stop training at {}".format(get_time_str())
            self.print_verbose(time_messgae)
            if self._message_recorder is not None:
                self._message_recorder.notify(time_messgae)

        all_callbacks.on_work_end()

    def _after_work(self):
        if self._checkpoint is not None:
            self.result = self._checkpoint.record
            self._best_result =  self._checkpoint.best_result
            self._best_epoch =  self._checkpoint.best_epoch


class TestWorker(Worker):
    """a worker which will evaluate the model"""
    def __init__(self,model,data,executor=None,generator=None,callbacks=None,path=None):
        super().__init__(model,data,executor,path=path)
        self._generator = generator
        self._callbacks = callbacks
        self._checkpoint = None

        if generator is None:
            self._generator = DataGenerator(shuffle=False)
        if callbacks is None:
            self._callbacks = Callbacks()

    def _save_setting(self):
        if self.path is not None:
            for key in ['_callbacks','executor','path']:
                attr = getattr(self,key)
                if hasattr(attr,'get_config'):
                    attr = attr.get_config()
                self._settings[key] = attr
                
            path = os.path.join(self.path,"test_worker_setting.json")
            self._settings['executor']['optimizer'] = self._settings['executor']['optimizer']['param_groups']
            write_json(self._settings,path)
            
    def _before_work(self):
        self._loader = self._generator(SeqDataset(self.data['testing']))
        record_path = None
        if self.path is not None:   
            record_path = os.path.join(self.path,"test_record.json")
            self._checkpoint = build_checkpoint(self.path,record_path,
                                                only_recorder=True,
                                                force_reset=True) 
        self._callbacks.add(DataHolder(prefix='test'))
        self._save_setting()

    def _work(self):
        """Test model"""
        epoch_start=1
        callbacks = Callbacks(self._callbacks)
        if self._checkpoint is not None:
            self._checkpoint.on_work_begin(worker=self)
        callbacks.on_work_begin(worker=self)
        record = {}
        if self._checkpoint is not None:
            self._checkpoint.on_epoch_begin(counter=epoch_start)
        callbacks.on_epoch_begin(counter=epoch_start)
        batch_info = "Testing data: {0:.1f}%"
        save_distribution = self.model.save_distribution
        self.model.save_distribution=False
        for index,item in enumerate(self._loader):
            seq_data = SeqDataset(item)
            evaluate_per_batch(self.model,seq_data,self.executor,self._callbacks)
            status = 100*index/len(self._loader)
            self.print_verbose(batch_info.format(status),True)

        self.model.save_distribution = save_distribution
        record = callbacks.get_data()
        if self._checkpoint is not None:
            self._checkpoint.on_epoch_end(metric=record)
        self.executor.on_epoch_end(epoch=epoch_start,metric=record)
        callbacks.on_epoch_end(metric=record)
        callbacks.on_work_end()
        if self._checkpoint is not None:
            self._checkpoint.on_work_end()

    def _after_work(self):
        if self._checkpoint is not None:
            if hasattr(self._checkpoint,'record'):
                self.result = self._checkpoint.record
            else:
                self.result = self._checkpoint.data
