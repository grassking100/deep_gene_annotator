"""This submodule provides trainer to train model"""
import warnings
import os,sys
import time
import json
import torch
from ..process.worker import Worker
from ..process.data_generator import DataGenerator
from .executer import BasicExecutor
from .callback import Accumulator, Recorder, Callbacks
from .warning import WorkerProtectedWarning

def train_per_batch(model,ids,inputs,labels,lengths,mask,executor,callbacks):
    callbacks.on_batch_begin()
    torch.cuda.empty_cache()
    inputs = torch.from_numpy(inputs).float().cuda()
    labels = torch.from_numpy(labels).long().cuda()
    mask = torch.from_numpy(mask).cuda()
    metric,outputs = executor.fit(model,inputs,labels,mask=mask,lengths=lengths)
    callbacks.on_batch_end(outputs=outputs,
                           labels=labels,
                           lengths=lengths,
                           metric=metric,
                           mask=mask,
                           ids=ids)

def evaluate_per_batch(model,ids,inputs,labels,lengths,mask,executor,callbacks):
    callbacks.on_batch_begin()
    torch.cuda.empty_cache()
    inputs = torch.from_numpy(inputs).float().cuda()
    labels = torch.from_numpy(labels).long().cuda()
    mask = torch.from_numpy(mask).cuda()
    metric,outputs = executor.evaluate(model,inputs,labels,mask=mask,lengths=lengths)
    callbacks.on_batch_end(outputs=outputs,
                           labels=labels,
                           lengths=lengths,
                           metric=metric,
                           mask=mask,
                           ids=ids)


class PyWorker(Worker):
    def __init__(self,executor=None):
        super().__init__()
        self.executor = executor
        if executor is None:
            self.executor = BasicExecutor()
            
    def print_verbose(self,info):
        if self.is_verbose_visible:
            print(info,end='\r')
            sys.stdout.write('\033[K')

def _init_generator(generator):
    generator.return_extra_info = True
    generator.order = 'NCL'
    generator.order_target=['answers','inputs']
    generator.pad_value={'answers':0,'inputs':0}

class TrainWorker(PyWorker):
    """a worker which will train and evaluate the model"""
    def __init__(self,executor=None,train_generator=None,val_generator=None,
                 train_callbacks=None,val_callbacks=None,other_callbacks=None,
                 writer=None,epoch_start=None,epoch=None):
        super().__init__(executor)
        self._train_generator = train_generator
        self._val_generator = val_generator
        self._train_callbacks = train_callbacks
        self._val_callbacks = val_callbacks
        self._other_callbacks = other_callbacks

        if train_generator is None:
            self._train_generator = DataGenerator()
        if val_generator is None:
            self._val_generator = DataGenerator()
        if train_callbacks is None:
            self._train_callbacks = Callbacks()
        if val_callbacks is None:
            self._val_callbacks = Callbacks()
        if other_callbacks is None:
            self._other_callbacks = Callbacks()

        self._writer = writer
        self._epoch_start = epoch_start or 0
        self._epoch = epoch or 1
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

    def before_work(self,path=None,**kwargs):
        self._train_generator.x_data = self.data['training']['inputs']
        self._train_generator.y_data = self.data['training']['answers']
        self._train_generator.extra = self.data['training']['extra']
        if 'validation' in self.data.keys():
            self._val_generator.x_data = self.data['validation']['inputs']
            self._val_generator.y_data = self.data['validation']['answers']
            self._val_generator.extra = self.data['validation']['extra']
            acc = Accumulator(prefix='val')
            self._val_callbacks.add(acc)
        acc = Accumulator()
        self._train_callbacks.add(acc)
        self._recoder = Recorder()
        self.executor.process(self.model)
        _init_generator(self._train_generator)
        _init_generator(self._val_generator)
        self._is_running = True

        if path is not None:
            for key in ['_train_callbacks','_val_callbacks','_other_callbacks',
                        '_epoch_start','_epoch','executor']:
                attr = getattr(self,key)
                if hasattr(attr,'get_config'):
                    attr = attr.get_config()
                self._settings[key] = attr

            with open(os.path.join(path,"train_worker_setting.json"),'w') as fp:
                json.dump(self._settings,fp, indent=4)

            self._recoder.path = os.path.join(path,'train_record.json')

    def work(self):
        """Train model"""
        self._validate()
        callbacks = Callbacks(self._train_callbacks)
        callbacks.add(self._val_callbacks)
        callbacks.add(self._other_callbacks)
        all_callbacks = Callbacks(callbacks)
        all_callbacks.add(self._recoder)
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

            for index,item in enumerate(self._train_generator):
                inputs, labels, extra = item
                ids,lengths,mask = extra['ids'],extra['lengths'],extra['mask']
                train_per_batch(self.model,ids,inputs,labels,lengths,mask,
                                self.executor,self._train_callbacks)
                status = 100*index/len(self._train_generator)
                self.print_verbose(batch_info.format(epoch,self._epoch,'training',status))

            for index,item in enumerate(self._val_generator):
                inputs, labels, extra = item
                ids,lengths,mask = extra['ids'],extra['lengths'],extra['mask']
                evaluate_per_batch(self.model,ids,inputs,labels,lengths,mask,
                                   self.executor,self._val_callbacks)
                status = 100*index/len(self._val_generator)
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
            self._train_generator.on_epoch_end()
            self._val_generator.on_epoch_end()
            self._train_callbacks.on_epoch_end(metric=train_record)
            self._val_callbacks.on_epoch_end(metric=val_record)
            self._other_callbacks.on_epoch_end(metric=record)
            self._recoder.on_epoch_end(metric=record)

            time_cost = round(time.time()-pre_time,3)
            self.print_verbose(epoch_info.format(epoch,self._epoch,time_cost,record))

            if not self.is_running:
                self.print_verbose("Early stop at {}".format(epoch))
                break

        all_callbacks.on_work_end()

    def after_work(self,**kwargs):
        self.result = self._recoder.data
        if self.best_epoch is not None:
            self._best_result = {}
            for key,value in self.result.items():
                self._best_result[key] = self.result[key][self.best_epoch - 1]

class TestWorker(PyWorker):
    """a worker which will train and evaluate the model"""
    def __init__(self,executor=None,generator=None,callbacks=None):
        super().__init__(executor)
        self._generator = generator
        self._callbacks = callbacks
        self._recoder = None

        if generator is None:
            self._generator = DataGenerator()
        if callbacks is None:
            self._callbacks = Callbacks()

    def before_work(self,path=None,**kwargs):
        item = self.data['testing']
        self._generator.x_data = item['inputs']
        self._generator.y_data = item['answers']
        self._generator.extra = item['extra']
        accum = Accumulator(prefix='test')
        self._callbacks.add(accum)
        self._recoder = Recorder()
        self.executor.process(self.model)
        if path is not None:
            for key in ['_callbacks']:
                attr = getattr(self,key)
                if hasattr(attr,'get_config'):
                    attr = attr.get_config()
                self._settings[key] = attr
            with open(os.path.join(path,"test_worker_setting.json"),'w') as fp:
                json.dump(self._settings,fp, indent=4)
            self._recoder.path = os.path.join(path,'test_record.json')


    def work(self):
        """Test model"""
        self._validate()
        callbacks = Callbacks(self._callbacks)
        self._recoder.on_work_begin(worker=self)
        callbacks.on_work_begin(worker=self)
        record = {}
        self._recoder.on_epoch_begin()
        callbacks.on_epoch_begin(counter=1)
        _init_generator(self._generator)
        batch_info = "Testing data: {0:.1f}%"

        for index,item in enumerate(self._generator):
            inputs, labels, extra = item
            ids,lengths,mask = extra['ids'],extra['lengths'],extra['mask']
            evaluate_per_batch(self.model,ids,inputs,labels,lengths,mask,
                               self.executor,self._callbacks)
            status = 100*index/len(self._generator)
            self.print_verbose(batch_info.format(status))

        record = callbacks.get_data()
        self._generator.on_epoch_end()
        self._recoder.on_epoch_end(metric=record)
        callbacks.on_epoch_end(metric=record)
        callbacks.on_work_end()
        self._recoder.on_work_end()

    def after_work(self,**kwargs):
        self.result = self._recoder.data
