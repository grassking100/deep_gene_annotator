"""This submodule provides trainer to train model"""
import torch
from ..process.worker import Worker
from ..pytorch.callback import Accumulator, Recorder
from ..function.data_generator import SeqGenerator

import matplotlib.pyplot as plt
import numpy as np
import time

def plt_dynamic(fig,ax,x,ys):
    for y in ys:
        ax.plot(x, y)
    fig.canvas.draw()
    
def _convert(inputs,labels):
    inputs = torch.from_numpy(inputs).type('torch.FloatTensor').cuda()
    labels = torch.from_numpy(labels).type('torch.LongTensor').cuda()
    return inputs,labels

def train(model,inputs,labels,compiler,lengths):
    model.train(True)
    compiler.optimizer.zero_grad()
    outputs = model(inputs,lengths=lengths)
    outputs = outputs.contiguous()
    outputs = outputs.view(-1, outputs.size()[1])
    loss_ = compiler.loss(outputs, labels)
    loss_.backward()
    compiler.optimizer.step()    
    return loss_.item(),outputs

def validate(model,inputs,labels,compiler,lengths):
    model.train(False)
    outputs = model(inputs,lengths=lengths)
    outputs = outputs.contiguous()
    outputs = outputs.view(-1, outputs.size()[1])
    loss_ = compiler.loss(outputs, labels)
    return loss_.item(),outputs

def fit_wrapper(epoch_num,batch_size=32,epoch_shuffle=False,
                train_callbacks=[],val_callbacks=[],*args,**argws):
    def fit(model,data,compiler):
        fig,ax = plt.subplots(1,1)
        data_ = {}
        history = Recorder()
        for data_kind,item in data.items():
            data_[data_kind] = SeqGenerator(item['inputs'],item['answers'],
                                            batch_size,epoch_shuffle=epoch_shuffle,
                                            return_len=True,order='NCL',padding_value=-1,
                                            answer_one_hot=False)
        train_loss = Accumulator(name='loss')
        train_loss.on_work_begin()
        has_valid = 'validation' in data_.keys()
        if has_valid:
            val_loss = Accumulator(prefix='val',name='loss')
            val_loss.on_work_begin()
        for callback in train_callbacks+val_callbacks:
            callback.on_work_begin()
        history.on_work_begin()
        for epoch in range(epoch_num):
            record = {}
            history.on_epoch_begin()
            for callback in train_callbacks+val_callbacks:
                callback.on_epoch_begin()
            train_loss.on_epoch_begin()
            for item in data_['training']:
                for callback in train_callbacks:
                    callback.on_batch_begin()
                train_loss.on_batch_begin()
                inputs, labels, lengths = item
                inputs,labels = _convert(inputs,labels)
                labels = labels.view(-1)
                loss_,outputs = train(model,inputs,labels,compiler,lengths)
                train_loss.on_batch_end(loss_)
                for callback in train_callbacks:
                    callback.on_batch_end(outputs,labels)
            train_loss.on_epoch_end()
            if hasattr(train_loss,'data'):
                for type_,value in train_loss.data.items():
                    record[type_]=value
            if has_valid:
                val_loss.on_epoch_begin()
                for item in data_['validation']:
                    for callback in val_callbacks:
                        callback.on_batch_begin()
                    val_loss.on_batch_begin()
                    inputs, labels, lengths = item
                    inputs,labels = _convert(inputs,labels)
                    labels = labels.view(-1)
                    loss_,outputs = validate(model,inputs,labels,compiler,lengths)
                    val_loss.on_batch_end(loss_)
                    for callback in val_callbacks:
                        callback.on_batch_end(outputs,labels)
                val_loss.on_epoch_end()
                if hasattr(val_loss,'data'):
                    for type_,value in val_loss.data.items():
                        record[type_]=value
            for callback in train_callbacks+val_callbacks:
                if hasattr(callback,'data'):
                    for type_,value in callback.data.items():
                        record[type_]=value
            history.on_epoch_end(record)
            #plt_dynamic(fig,ax, np.arange(epoch+1), [history.data['loss'],history.data['val_loss']])
            for callback in train_callbacks+val_callbacks:
                callback.on_epoch_end()
            print(epoch + 1, record)
        for callback in train_callbacks+val_callbacks+[history]:
            callback.on_work_end()
    return fit

class TrainWorker(Worker):
    """a worker which will train and evaluate the model"""
    def __init__(self,wrapper,path_root=None,is_verbose_visible=True):
        super().__init__(wrapper,path_root)
        self._result = {}
        self.is_verbose_visible = is_verbose_visible

    def before_work(self):
        if self._path_root is not None:
            root_path = './'+self._path_root
            create_folder(root_path+"/model")
            create_folder(root_path+"/log")
            create_folder(root_path+"/result")
            plot_saved_path = root_path+"/model_image.png"
            #plot_model(self.model,show_shapes=True,to_file=plot_saved_path)
            length = str(len(str(self._epoch)))
            model_saved_path = root_path+'/model/epoch_{:0'+length+'d}.h5'
            model_saved_path = model_saved_path.format(0)
            #self.model.save(model_saved_path)

    def work(self):
        """Train model"""
        self._validate()
        self._wrapper(self.model,self.data,self.compiler)

    def after_work(self):
        """Do something after worker work"""
        pass