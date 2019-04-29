"""This submodule provides trainer to train model"""
import torch
from ..process.worker import Worker
from ..pytorch.callback import Accumulator, Recorder
from ..process.data_generator import DataGenerator
import numpy as np
import time
import json

def train_per_batch(model,inputs,labels,extra,compiler,callbacks):
    for callback in callbacks:
        callback.on_batch_begin()
    torch.cuda.empty_cache()
    inputs = torch.from_numpy(inputs).float().cuda()
    labels = torch.from_numpy(labels).long().cuda()
    loss_ = compiler.fit(model,inputs,labels,**extra)
    for callback in callbacks:
        callback.on_batch_end(outputs=model.saved_outputs,
                              index_outputs=model.saved_index_outputs,
                              labels=labels,
                              lengths=model.saved_lengths,
                              metric=loss_,
                              ids=extra['ids'],
                              parmeters=model.named_parameters())

def evaluate_per_batch(model,inputs,labels,extra,compiler,callbacks):
    for callback in callbacks:
        callback.on_batch_begin()
    torch.cuda.empty_cache()
    inputs = torch.from_numpy(inputs).float().cuda()
    labels = torch.from_numpy(labels).long().cuda()
    loss_ = compiler.evaluate(model,inputs,labels,**extra)
    for callback in callbacks:
        callback.on_batch_end(outputs=model.saved_outputs,
                              index_outputs=model.saved_index_outputs,
                              labels=labels,
                              lengths=model.saved_lengths,
                              metric=loss_,
                              ids=extra['ids'],
                              parmeters=model.named_parameters())

class PyWorker(Worker):
    def __init__(self):
        super().__init__()
        self._settings = locals()
        self._data = {}
        
    def _validate(self):
        """Validate required data"""
        super()._validate()
        if self.compiler is None:
            raise Exception("Compiler must be setted")

class TrainWorker(PyWorker):
    """a worker which will train and evaluate the model"""
    def __init__(self,train_generator=None,val_generator=None):
        super().__init__()
        self._train_generator = train_generator
        self._val_generator = val_generator
        self.train_callbacks = []
        self.val_callbacks = []
        self.other_callbacks = []
        self.writer = None
        self.epoch_start = 0
        self.epoch_num = 1
        self.is_running = True
        if self._train_generator is None:
            self._train_generator = DataGenerator()
        if self._train_generator is None:
            self._val_generator = DataGenerator()

    def before_work(self,path=None):
        self._train_generator.x_data = self.data['training']['inputs']
        self._train_generator.y_data = self.data['training']['answers']
        self._train_generator.extra = self.data['training']['extra']
        if 'validation' in self.data.keys():
            self._val_generator.x_data = self.data['validation']['inputs']
            self._val_generator.y_data = self.data['validation']['answers']
            self._val_generator.extra = self.data['validation']['extra']
            acc = Accumulator()
            acc.prefix='val'
            acc.name='loss'
            self.val_callbacks.append(acc)
        acc = Accumulator()
        acc.name='loss'
        self.train_callbacks.append(acc)
        record_saved_path = None
        if path is not None:
            for key in ['train_callbacks','val_callbacks','other_callbacks','epoch_start','epoch_num']:
                self._settings[key] = getattr(self,key)
            record_saved_path = path+'/train_record.json'
            json_path = path + "/train_worker_setting.json"
            with open(json_path,'w') as fp:
                try:
                    json.dump(self._settings,fp)
                except TypeError:
                     fp.write(str(self._settings))
            self._recoder = Recorder()
            self._recoder.path=record_saved_path
    def work(self):
        """Train model"""
        self._validate()
        callbacks = self.train_callbacks+self.val_callbacks + self.other_callbacks
        all_callbacks = callbacks + [self._recoder]
        for callback in all_callbacks:
            callback.on_work_begin(worker=self)
        for epoch in range(self.epoch_start+1,self.epoch_start+1+self.epoch_num):
            pre_time = time.time()
            self.writer.counter = epoch
            for callback in all_callbacks:
                callback.on_epoch_begin(counter=epoch)
            for item in self._train_generator:
                inputs, labels, extra = item
                train_per_batch(self.model,inputs,labels,extra,
                                self.compiler,self.train_callbacks)
            for item in self._val_generator:
                inputs, labels, extra = item
                evaluate_per_batch(self.model,inputs,labels,extra,
                                   self.compiler,self.val_callbacks)
            record = {}
            for callback in callbacks:
                if hasattr(callback,'data'):
                    for type_,value in callback.data.items():
                        record[type_]=value
            if record['loss'] is None or record['val_loss'] is None:
                self.is_running = False
            self._train_generator.on_epoch_end()
            self._val_generator.on_epoch_end()
            for callback in all_callbacks:
                callback.on_epoch_end(metric=record)
            if self.is_verbose_visible:
                time_cost = time.time()-pre_time
                print(epoch ,time_cost, record)
            if not self.is_running:
                print("Early stop at "+str(epoch))
                break
        for callback in all_callbacks:
            callback.on_work_end()
        self.result = self._recoder.data

class TestWorker(PyWorker):
    """a worker which will train and evaluate the model"""
    def __init__(self,generator=None):
        super().__init__()
        self._generator = generator
        self.callbacks = []
        if self._generator is None:
            self._generator = DataGenerator()

    def before_work(self,path=None):
        item = self.data['testing']
        self._generator.x_data = item['inputs']
        self._generator.y_data = item['answers']
        self._generator.extra = item['extra']
        self._callbacks.append(Accumulator(prefix='test',name='loss'))
        record_saved_path = None
        if path is not None:
            record_saved_path = path+'/test_record.json'
            json_path = path + "/test_worker_setting.json"
            for key in ['callbacks']:
                self._settings[key] = getattr(self,key)
            with open(json_path,'w') as fp:
                try:
                    json.dump(self._settings,fp)
                except TypeError:
                     fp.write(str(self._settings))
        self._recoder = Recorder()
        self._recoder.path=record_saved_path
    def work(self):
        """Test model"""
        self._validate()
        callbacks = self.callbacks + [self._recoder]
        for callback in callbacks:
            callback.on_work_begin(model=self.model)
        record = {}
        for callback in callbacks:
            callback.on_epoch_begin(worker=self)
        for item in self._generator:
            inputs, labels, extra = item
            evaluate_per_batch(self.model,inputs,labels,extra,
                               self.compiler,self.callbacks)
        for callback in self.callbacks:
            if hasattr(callback,'data'):
                for type_,value in callback.data.items():
                    record[type_]=value
        self._generator.on_epoch_end()
        for callback in callbacks:
            callback.on_epoch_end(metric=record)
        for callback in callbacks:
            callback.on_work_end()
        self.result = self._recoder.data