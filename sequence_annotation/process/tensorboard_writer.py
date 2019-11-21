import torch
from  matplotlib import pyplot
import numpy as np

class TensorboardWriter:
    def __init__(self,writer):
        self._writer = writer
        self.counter = 0

    def close(self):
        self._writer.close()
        
    def add_scalar(self,record,prefix=None,counter=None):
        prefix = prefix or ''
        for name,val in record.items():
            val = np.array(val).reshape(-1)
            if 'val' in name:
                name = '_'.join(name.split('_')[1:])
            if len(val)==1:
                self._writer.add_scalar(prefix+name, val, counter)

    def add_grad(self,named_parameters,prefix=None,counter=None):
        prefix = prefix or ''
        counter = counter or self.counter
        for name,param in named_parameters:
            if param.grad is not None:
                grad = param.grad.cpu().detach().numpy()
                if  np.isnan(grad).any():
                    print(grad)
                    raise Exception(name+" has at least one NaN in it.")
                self._writer.add_histogram("layer_"+prefix+'grad_'+name, grad, counter)

    def add_distribution(self,name,data,prefix=None,counter=None):
        prefix = prefix or ''
        counter = counter or self.counter
        try:
            values = data.contiguous().view(-1).cpu().detach().numpy()
        except:
            raise Exception("{} causes something wrong occur".format(name))
        if  np.isnan(values).any():
            print(values)
            raise Exception(name+" has at least one NaN in it.")
        self._writer.add_histogram(prefix+name, values,counter)

    def add_weights(self,named_parameters,prefix=None,counter=None):
        prefix = prefix or ''
        counter = counter or self.counter
        for name,param in named_parameters:
            w = param.cpu().detach().numpy()
            if  np.isnan(w).any():
                print(w)
                raise Exception(name+" has at least one NaN in it.")
            self._writer.add_histogram("layer_"+prefix+name, w, counter)

    def add_figure(self,name,value,prefix=None,counter=None,title='',labels=None,
                   colors=None,use_stack=False,*args,**kwargs):
        prefix = prefix or ''
        counter = counter or self.counter
        #data shape is L,C
        if len(value.shape)!=2:
            raise Exception("Value shape size should be two",value.shape)
        fig = pyplot.figure(dpi=200)
        if isinstance(value,torch.Tensor):
            value = value.cpu().detach().numpy()
        if  np.isnan(value).any():
            raise Exception(title+" has at least one NaN in it.")
        length = value.shape[0]
        if labels is not None:
            value = value.transpose()
            if len(value) != len(labels):
                raise Exception("Labels' size({}) is not same as data's size({})".format(len(labels),len(value)))
            if colors is not None:
                if len(colors) != len(labels):
                    raise Exception("Labels' size({}) is not same as colors's size({})".format(len(labels),len(colors)))    
            if use_stack:
                pyplot.stackplot(list(range(length)),value,labels=labels,colors=colors)
            else:
                for index in range(len(labels)):
                    item = value[index]
                    label = labels[index]
                    if colors is not None:
                        color = colors[index]
                    else:
                        color = None
                    pyplot.plot(item,label=label,color=color)
            pyplot.legend()
        else:
            if use_stack:
                pyplot.stackplot(list(range(length)),value)
            else:
                pyplot.plot(value)
        pyplot.xlabel("Sequence (from 5' to 3')")
        pyplot.ylabel("Value")
        pyplot.title(title)
        pyplot.close(fig)
        self._writer.add_figure(prefix+name,fig,global_step=counter,*args,**kwargs)

    def add_matshow(self,name,value,prefix=None,counter=None,title='',*args,**kwargs):
        prefix = prefix or ''
        counter = counter or self.counter
        fig = pyplot.figure(dpi=200)
        if  np.isnan(value).any():
            print(value)
            raise Exception(title+" has at least one NaN in it.")
        cax = pyplot.matshow(value,fignum=0)
        fig.colorbar(cax)
        pyplot.title(title)
        pyplot.close(fig)
        self._writer.add_figure(prefix+name,fig,global_step=counter,*args,**kwargs)