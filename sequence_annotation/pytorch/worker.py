"""This submodule provides trainer to train model"""
import torch
from ..process.worker import Worker
from ..pytorch.callback import Accumulator, Recorder
from ..function.data_generator import DataGenerator
import numpy as np
import time

class PyWorker(Worker):
    def __init__(self,generator=None,epoch_num=1,path_root=None,
                 writer=None):
        super().__init__(path_root=path_root)
        self._result = {}
        self._epoch_num = epoch_num
        self._generator = generator or DataGenerator
        self._writer = writer
        self._data = {}

    def _validate(self):
        """Validate required data"""
        super()._validate()
        if self.compiler is None:
            raise Exception("Compiler must be setted")

class Trainer(PyWorker):
    """a worker which will train and evaluate the model"""
    def __init__(self,generator=None,epoch_num=1,path_root=None,
                 train_callbacks=None,val_callbacks=None,other_callbacks=None,
                 writer=None,epoch_start=0,**kwargs):
        super().__init__(generator=generator,epoch_num=epoch_num,
                         path_root=path_root,writer=writer)
        self._train_callbacks = train_callbacks or []
        self._val_callbacks = val_callbacks or []
        self._other_callbacks = other_callbacks or []
        self._writer = writer
        self._kwargs = kwargs
        self._epoch_start = epoch_start
        self._RUNNING = True
    def before_work(self):
        if self._path_root is not None:
            root_path = './'+self._path_root
            create_folder(root_path+"/model")
            create_folder(root_path+"/log")
            create_folder(root_path+"/result")
            plot_saved_path = root_path+"/model_image.png"
            length = str(len(str(self._epoch)))
            model_saved_path = root_path+'/model/epoch_{:0'+length+'d}.h5'
            model_saved_path = model_saved_path.format(0)
        for data_kind in self.data.keys():
            item = self.data[data_kind]
            self._data[data_kind] = self._generator(item['inputs'],item['answers'],
                                                    extra=item['extra'],**self._kwargs)
        if 'validation' not in self._data.keys():
            self._data['validation'] = []
        else:
            self._val_callbacks.append(Accumulator(prefix='val',name='loss'))
        self._train_callbacks.append(Accumulator(name='loss'))
        self._result = Recorder()

    def work(self):
        """Train model"""
        self._validate()
        callbacks = self._train_callbacks+self._val_callbacks + [self._result] + self._other_callbacks
        for callback in callbacks:
            callback.on_work_begin(model=self.model)
        for epoch in range(self._epoch_start+1,self._epoch_start+1+self._epoch_num):
            pre_time = time.time()
            self._writer.counter = epoch
            for callback in callbacks:
                callback.on_epoch_begin()
            for item in self._data['training']:
                for callback in self._train_callbacks:
                    callback.on_batch_begin()
                inputs, labels, extra = item
                inputs = torch.from_numpy(inputs).float().cuda()
                labels = torch.from_numpy(labels).long().cuda()
                loss_ = self.compiler.fit(self.model,inputs,labels,**extra)

                for callback in self._train_callbacks:
                    callback.on_batch_end(outputs=self.model.saved_outputs,
                                          index_outputs=self.model.saved_index_outputs,
                                          labels=labels,
                                          lengths=self.model.saved_lengths,
                                          metric=loss_,
                                          ids=extra['ids'],
                                          parmeters=self.model.named_parameters())
            for item in self._data['validation']:
                for callback in self._val_callbacks:
                    callback.on_batch_begin()
                inputs, labels, extra = item
                inputs = torch.from_numpy(inputs).float().cuda()
                labels = torch.from_numpy(labels).long().cuda()
                loss_ = self.compiler.evaluate(self.model,inputs,labels,**extra)
                for callback in self._val_callbacks:
                    callback.on_batch_end(outputs=self.model.saved_outputs,
                                          index_outputs=self.model.saved_index_outputs,
                                          labels=labels,
                                          lengths=self.model.saved_lengths,
                                          metric=loss_,ids=extra['ids'],
                                          parmeters=self.model.named_parameters()
                                          )
            record = {}
            for callback in self._train_callbacks+self._val_callbacks:
                if hasattr(callback,'data'):
                    for type_,value in callback.data.items():
                        record[type_]=value
            self._data['training'].on_epoch_end()
            self._data['validation'].on_epoch_end()
            for callback in callbacks:
                callback.on_epoch_end(metric=record,worker=self)
            if self.is_verbose_visible:
                time_cost = time.time()-pre_time
                print(epoch ,time_cost, record)
            if not self._RUNNING:
                print("Early stop at "+str(epoch))
                break
        for callback in callbacks:
            callback.on_work_end()
        
    def after_work(self):
        if self._path_root is not None:
            data = json.loads(pd.Series(self._result).to_json(orient='index'))
            with open(self._path_root + '/result/record.json', 'w') as outfile:  
                json.dump(data, outfile,indent=4)
                
class Tester(PyWorker):
    """a worker which will train and evaluate the model"""
    def __init__(self,generator=None,epoch_num=1,path_root=None,
                 writer=None,callbacks=None,**kwargs):
        super().__init__(generator=generator,epoch_num=epoch_num,
                         path_root=path_root,writer=writer)
        self._callbacks = callbacks or []
        self._kwargs = kwargs
    def before_work(self):
        if self._path_root is not None:
            root_path = './'+self._path_root
            create_folder(root_path+"/log")
            create_folder(root_path+"/test")
        item = self.data['testing']
        self._data = self._generator(item['inputs'],item['answers'],extra=item['extra'],
                                     **self._kwargs)
        self._callbacks.append(Accumulator(prefix='test',name='loss'))
        self._result = Recorder()

    def work(self):
        """Test model"""
        self._validate()
        callbacks = self._callbacks + [self._result]
        for callback in callbacks:
            callback.on_work_begin(model=self.model)
        record = {}
        for callback in callbacks:
            callback.on_epoch_begin()
        for item in self._data:
            for callback in self._callbacks:
                callback.on_batch_begin()
            inputs, labels, extra = item
            inputs = torch.from_numpy(inputs).float().cuda()
            labels = torch.from_numpy(labels).long().cuda()
            loss_ = self.compiler.evaluate(self.model,inputs,labels,lengths=lengths,**extra)
            for callback in self._callbacks:
                callback.on_batch_end(outputs=self.model.saved_outputs,labels=labels,
                                      metric=loss_)
        for callback in self._callbacks:
            if hasattr(callback,'data'):
                for type_,value in callback.data.items():
                    record[type_]=value
        self._data.on_epoch_end()
        for callback in callbacks:
            callback.on_epoch_end(metric=record)
        if self.is_verbose_visible:
            print(epoch , record)
        for callback in callbacks:
            callback.on_work_end()
        
    def after_work(self):
        if self._path_root is not None:
            data = json.loads(pd.Series(self._result).to_json(orient='index'))
            with open(self._path_root + '/test/evaluate.json', 'w') as outfile:  
                json.dump(data, outfile,indent=4)
            