import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
from ..utils.utils import CONSTANT_DICT
from ..file_process.utils import BASIC_GENE_ANN_TYPES,EXON_TYPE,INTRON_TYPE,INTERGENIC_REGION_TYPE
from .callback import Callback,get_prefix
from .data_generator import SeqDataset
from .executor import predict

pyplot.switch_backend('agg')

#BASIC_SIMPLIFY_MAP = CONSTANT_DICT({
#    'exon': ['exon'],
#    'intron': ['intron'],
#    'other': ['other']
#})


BASIC_COLOR_SETTING = CONSTANT_DICT({
    INTERGENIC_REGION_TYPE: 'blue',
    EXON_TYPE: 'red',
    INTRON_TYPE: 'yellow'
})


class TensorboardWriter:
    def __init__(self, writer_or_root):
        if isinstance(writer_or_root, str):
            self._writer = SummaryWriter(writer_or_root)
        else:
            self._writer = writer_or_root
        self.counter = 0

    def close(self):
        if not isinstance(self._writer, str):
            self._writer.close()

    def add_scalar(self, record, prefix=None, counter=None):
        prefix = prefix or ''
        for name, val in record.items():
            val = np.array(val).reshape(-1)
            if 'val' in name:
                name = '_'.join(name.split('_')[1:])
            if len(val) == 1:
                self._writer.add_scalar(prefix + name, val, counter)

    def add_grad(self, named_parameters, prefix=None, counter=None):
        prefix = prefix or ''
        counter = counter or self.counter
        for name, param in named_parameters:
            if param.grad is not None:
                grad = param.grad.cpu().detach().numpy()
                if np.isnan(grad).any():
                    print(grad)
                    raise Exception(name + " has at least one NaN in it.")
                self._writer.add_histogram("layer_" + prefix + 'grad_' + name,
                                          grad, counter)

    def add_distribution(self, name, data, prefix=None, counter=None):
        prefix = prefix or ''
        counter = counter or self.counter
        if isinstance(data, torch.Tensor):
            try:
                data = data.cpu().detach().numpy()
            except BaseException:
                print(data)
                raise Exception("{} causes something wrong occur".format(name))
        if np.isnan(data).any():
            print(data)
            raise Exception(name + " has at least one NaN in it.")
        try:
            self._writer.add_histogram(prefix + name, data.flatten(), counter)
        except BaseException:
            print(data)
            raise Exception("{} causes something wrong occur".format(name))

    def add_weights(self, named_parameters, prefix=None, counter=None):
        prefix = prefix or ''
        counter = counter or self.counter
        for name, param in named_parameters:
            w = param.cpu().detach().numpy()
            if np.isnan(w).any():
                print(w)
                raise Exception(name + " has at least one NaN in it.")
            self._writer.add_histogram("layer_" + prefix + name, w, counter)

    def add_figure(self,name,value,prefix=None,counter=None,title=None,
                   labels=None,colors=None,use_stack=False,*args,**kwargs):
        title = title or 'It'
        prefix = prefix or ''
        counter = counter or self.counter
        # data shape is L,C
        if len(value.shape) != 2:
            raise Exception("Value shape size should be two", value.shape)
        fig = pyplot.figure(dpi=200)
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        if np.isnan(value).any():
            raise Exception(title + " has at least one NaN in it.")
        length = value.shape[0]
        if labels is not None:
            value = value.transpose()
            if len(value) != len(labels):
                raise Exception("Labels' size({}) is not same as data's size({})".format(
                    len(labels), len(value)))
            if colors is not None:
                if len(colors) != len(labels):
                    raise Exception("Labels' size({}) is not same as colors's size({})".format(
                        len(labels), len(colors)))
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
                    pyplot.plot(item, label=label, color=color)
            pyplot.legend()
        else:
            if use_stack:
                pyplot.stackplot(list(range(length)), value)
            else:
                pyplot.plot(value)
        pyplot.xlabel("Sequence (from 5' to 3')")
        pyplot.ylabel("Value")
        pyplot.title(title)
        pyplot.close(fig)
        self._writer.add_figure(prefix + name,fig,global_step=counter,*args,**kwargs)

    def add_matshow(self,name,value,prefix=None,counter=None,
                    title=None,*args,**kwargs):
        title = title or 'It'
        prefix = prefix or ''
        counter = counter or self.counter
        fig = pyplot.figure(dpi=200)
        if np.isnan(value).any():
            print(value)
            raise Exception(title + " has at least one NaN in it.")
        cax = pyplot.matshow(value, fignum=0)
        fig.colorbar(cax)
        pyplot.title(title)
        pyplot.close(fig)
        self._writer.add_figure(prefix + name,fig,global_step=counter,*args,**kwargs)

        
class TensorboardCallback(Callback):
    def __init__(self, writer_or_root, prefix=None):
        self._prefix = get_prefix(prefix)
        if isinstance(writer_or_root,str):
            self.tensorboard_writer = TensorboardWriter(writer_or_root)
        else:
            self.tensorboard_writer = writer_or_root
        #self._model = None
        self._counter = None
        #self.do_add_grad = False
        #self.do_add_weights = False
        self.do_add_scalar = True
        
    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['prefix'] = self._prefix
        #config['do_add_grad'] = self.do_add_grad
        #config['do_add_weights'] = self.do_add_weights
        config['do_add_scalar'] = self.do_add_scalar
        return config
    
    #def on_work_begin(self, worker, **kwargs):
        #self._model = worker.executor.model

    def on_epoch_begin(self, counter, **kwargs):
        self._counter = counter

    def on_epoch_end(self, metric, **kwargs):
        #if self.do_add_grad:
        #    self.tensorboard_writer.add_grad(
        #        counter=self._counter,
        #        named_parameters=self._model.named_parameters(),
        #        prefix=self._prefix)
        #if self.do_add_weights:
        #    self.tensorboard_writer.add_weights(
        #        counter=self._counter,
        #        named_parameters=self._model.named_parameters(),
        #        prefix=self._prefix)
        if self.do_add_scalar:
            self.tensorboard_writer.add_scalar(counter=self._counter,
                                               record=metric,
                                               prefix=self._prefix)

    def on_work_end(self):
        self.tensorboard_writer.close()


class SeqFigCallback(Callback):
    def __init__(self,model,data,writer_or_root,
                 label_names=None,color_settings=None,prefix=None):
        self._prefix = get_prefix(prefix)
        if isinstance(writer_or_root,str):
            self._writer = TensorboardWriter(writer_or_root)
        else:
            self._writer = BASIC_GENE_ANN_TYPES
        self._data = SeqDataset({'input':data['input'],'length':[data['input'].shape[2]]})
        self._ann_vec = data['answer']
        self._counter = None
        self._model = model
        if label_names is None:
            label_names=BASIC_GENE_ANN_TYPES,
        if color_settings is None:
            color_settings=BASIC_COLOR_SETTING
        self.label_names = label_names
        self.colors = [color_settings[type_] for type_ in label_names]
        self.do_add_distribution = True
        if len(data) != 1 or len(answer) != 1:
            raise Exception("Data size should be one,", data.shape,answer.shape)
            
    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config['prefix'] = self._prefix
        config['label_names'] = self.label_names
        config['colors'] = self.colors
        return config
    
    def on_epoch_begin(self, counter, **kwargs):
        self._counter = counter
        if self._counter == 1:
            self._writer.add_figure("answer_figure",
                                    self._answer[0],
                                    prefix=self._prefix,
                                    colors=self.colors,
                                    labels=self.label_names,
                                    title="Answer figure",
                                    use_stack=True)


    def on_epoch_end(self, **kwargs):
        # Value's shape should be (1,C,L)
        self._model.reset()
        result = predict(self._model,self._data,inference=self._inference)['predict']
        result = result.get('annotation').cpu().numpy()[0]
        result = np.transpose(result)
        L, C = result.shape
        self._model.reset()
        torch.cuda.empty_cache()
        if self.do_add_distribution and hasattr(self._model,'saved_distribution'):
            for name, value in self._model.saved_distribution.items():
                self._writer.add_distribution(name,value,prefix=self._prefix,
                                              counter=self._counter)
                if len(value.shape) == 3:
                    value = value[0]
                value = value.transpose()
                self._writer.add_figure(name + "_figure",value,prefix=self._prefix,
                                        counter=self._counter,title=name)
        self._writer.add_figure("result_figure",result,prefix=self._prefix,
                                colors=self.colors,labels=self.label_names,
                                title="Result figure",use_stack=True)
        diff = np.transpose(result) - self._answer[0][:L, :]
        self._writer.add_figure("diff_figure",diff,prefix=self._prefix,
                                colors=self.colors,labels=self.label_names,
                                title="Predict - Answer figure",use_stack=False)
        
