from abc import ABCMeta, abstractmethod, abstractproperty
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import torch
from  matplotlib import pyplot
import numpy as np
import os
from ..genome_handler.region_extractor import GeneInfoExtractor
from ..genome_handler.seq_container import SeqInfoContainer
from ..genome_handler.ann_seq_processor import vecs2seq
from ..utils.utils import index2onehot

class TensorboardWriter:
    def __init__(self,writer):
        self._writer = writer
        self.counter = 0

    def add_scalar(self,record,prefix='',counter=None):
        for name,val in record.items():
            if 'val' in name:
                name = '_'.join(name.split('_')[1:])
            self._writer.add_scalar(prefix+name, np.array(val), counter)

    def add_grad(self,named_parameters,prefix='',counter=None):
        counter = counter or self.counter
        for name,param in named_parameters:
            if param.grad is not None:
                grad = param.grad.cpu().detach().numpy()
                if  np.isnan(grad).any():
                    print(grad)
                    raise Exception(name+" has at least one NaN in it.")
                self._writer.add_histogram("layer_"+prefix+'grad_'+name, grad, counter)

    def add_distribution(self,name,data,prefix='',counter=None):
        counter = counter or self.counter
        values = data.contiguous().view(-1).cpu().detach().numpy()
        if  np.isnan(values).any():
            print(values)
            raise Exception(name+" has at least one NaN in it.")
        self._writer.add_histogram(prefix+name, values,counter)

    def add_weights(self,named_parameters,prefix='',counter=None):
        counter = counter or self.counter
        for name,param in named_parameters:
            w = param.cpu().detach().numpy()
            if  np.isnan(w).any():
                print(w)
                raise Exception(name+" has at least one NaN in it.")
            self._writer.add_histogram("layer_"+prefix+name, w, counter)

    def add_figure(self,name,value,prefix='',counter=None,title='',labels=None,
                   colors=None,use_stack=False,*args,**kwargs):
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
                raise Exception("Labels' size is not same as data's size")
            if colors is not None:
                if len(colors) != len(labels):
                    raise Exception("Labels' size is not same as colors's size")    
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

    def add_matshow(self,name,value,prefix,counter=None,title='',*args,**kwargs):
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

class Callback(metaclass=ABCMeta):
    def on_work_begin(self,**kwargs):
        pass
    def on_work_end(self):
        pass
    def on_epoch_begin(self,**kwargs):
        pass
    def on_epoch_end(self,**kwargs):
        pass
    def on_batch_begin(self):
        pass
    def on_batch_end(self,**kwargs):
        pass
    
class Callbacks(Callback):
    def __init__(self,callbacks=None):
        self._callbacks = []
        if callbacks is not None:
            self.add_callbacks(callbacks)

    def clean(self):
        self._callbacks = []

    @property    
    def callbacks(self):
        return self._callbacks

    @callbacks.setter
    def callbacks(self,callbacks):
        self._callbacks = []
        self._callbacks = self.add_callbacks(callbacks)

    def add_callbacks(self,callbacks):
        list_ = []
        if isinstance(callbacks,Callbacks):
            for callback in callbacks.callbacks:
                if isinstance(callback,Callbacks):
                    list_ += callback.callbacks
                else:
                    list_ += [callback]
        else:
            list_ += [callbacks]
        self._callbacks += list_

    def on_work_begin(self,**kwargs):
        for callback in self.callbacks:
            callback.on_work_begin(**kwargs)

    def on_work_end(self):
        for callback in self.callbacks:
            callback.on_work_end()

    def on_epoch_begin(self,**kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(**kwargs)

    def on_epoch_end(self,**kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(**kwargs)

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self,**kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(**kwargs)

    def get_data(self):
        record = {}
        for callback in self.callbacks:
            if hasattr(callback,'data') and callback.data is not None:
                for type_,value in callback.data.items():
                    record[type_]=value
        return record

class GFFCompare(Callback):
    def __init__(self,ann_types,path,simplify_map):
        self.extractor = GeneInfoExtractor()
        self._path = path
        self._simplify_map = simplify_map
        self._ann_types = ann_types
        self.prefix = ""
        self._counter = 0

    def on_epoch_begin(self,counter,**kwargs):
        self._outputs = SeqInfoContainer()
        self._answers = SeqInfoContainer()
        self._counter = counter

    def on_batch_end(self,outputs,labels,ids,lengths,**kwargs):
        C = len(self._ann_types)
        indice = list(range(C))
        for id_,output, label, length in zip(ids,outputs,labels,lengths):
            output = output.max(0)[1][:length]
            output = index2onehot(output.cpu().numpy(),C)
            seq = vecs2seq(output,id_,'plus',self._ann_types)
            seq_informs = self.extractor.extract_per_seq(seq,self._simplify_map)
            self._outputs.add(seq_informs)
            if self._counter==1:
                label = label.transpose(0,1)[:length].transpose(0,1).cpu().numpy()
                ann = vecs2seq(label,id_,'plus',self._ann_types)
                ann_informs = self.extractor.extract_per_seq(ann,self._simplify_map)
                self._answers.add(ann_informs)

    def on_epoch_end(self,**kwargs):
        if self.prefix != "":
            path = self._path+"/"+self.prefix+"_gffcompare_"+str(self._counter)
        else:
            path = self._path+"/gffcompare_"+str(self._counter)
        predict_path = path+".gff3"
        answer_path = self._path+"/answers.gff3"
        if self._counter==1:
            self._answers.to_gff().to_csv(answer_path,index=None,header=None,sep='\t')
            os.system('python3 sequence_annotation/gene_info/script/python/gff2bed.py answers.gff3 answers.bed')
        if not self._outputs.is_empty():
            self._outputs.to_gff().to_csv(predict_path,index=None,header=None,sep='\t')
            os.system('~/../home/gffcompare/gffcompare -T -r '+answer_path+' '+ predict_path+' -o '+path)
            os.system('rm '+path+'.tracking')
            os.system('rm '+path+'.loci')
            os.system('python3 sequence_annotation/gene_info/script/python/gff2bed.py '+path+'.gff3 '+path+'.bed')
            
class EarlyStop(Callback):
    def __init__(self):
        self.target = 'val_loss'
        self.optimize_min = True
        self.patient = 16
        self.path = None
        self.period = None
        self.save_best_weights = False
        self.restore_best_weights = False
        self._counter = 0
        self._best_result = None
        self._best_epoch = 0
        self._model_weights = None
        self._worker = None

    @property
    def best_result(self):
        return self._best_result
    @property
    def best_epoch(self):
        return self._best_epoch
        
    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter
        if self.path is not None and self.period is not None:
            if (self._counter%self.period) == 0:
                model_path = self.path+'/model_epoch_'+str(self._counter)+'.pth'
                print("Save model at "+model_path)
                torch.save(self._worker.model.state_dict(),model_path)

    def on_epoch_end(self,metric,**kwargs):
        target = metric[self.target]
        update = False
        if self._best_result is None:
            update = True
        elif self.optimize_min:
            if self._best_result > target:
                update = True
        else:
            if self._best_result < target:
                update = True
        if update:
            if self.save_best_weights:
                self._model_weights = self._worker.model.state_dict()
            self._best_epoch = self._counter
            self._best_result = target
        if self.patient is not None:
            if (self._counter-self.best_epoch) > self.patient:
                self._worker.is_running = False

    def on_work_end(self):
        print("Best "+str(self.target)+": "+str(self._best_result))
        if self.save_best_weights and self.restore_best_weights:
            self._worker.model.load_state_dict(self._model_weights)
        if self.path is not None:
            model_path =  self.path+'/model_last_epoch_'+str(self._counter+1)+'.pth'
            print("Save model at "+model_path)
            torch.save(self._worker.model.state_dict(),model_path)

    def on_work_begin(self, worker,**kwargs):
        if self.save_best_weights:
            self._model_weights = worker.model.state_dict()
        self._worker = worker

class TensorboardCallback(Callback):
    def __init__(self,tensorboard_writer):
        self.tensorboard_writer = tensorboard_writer
        self._prefix = ""
        self._model = None
        self._counter = 0
        self.do_add_grad = False
        self.do_add_weights = False
        self.do_add_scalar = True
    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter
    @property
    def prefix(self):
        return self._prefix
    @prefix.setter
    def prefix(self,value=''):
        if value != '':
            value+="_"
        self._prefix = value
        
    def on_epoch_end(self,metric,**kwargs):
        if self.do_add_grad:
            self.tensorboard_writer.add_grad(counter=self._counter,
                                             named_parameters=self._model.named_parameters(),
                                             prefix=self.prefix)
        if self.do_add_weights:
            self.tensorboard_writer.add_weights(counter=self._counter,
                                                named_parameters=self._model.named_parameters(),
                                                prefix=self.prefix)
        if self.do_add_scalar:
            self.tensorboard_writer.add_scalar(counter=self._counter,record=metric,
                                               prefix=self.prefix)
    def on_work_begin(self,worker,**kwargs):
        self._model = worker.model

class SeqFigCallback(Callback):
    def __init__(self,tensorboard_writer,data,answer):
        self._writer = tensorboard_writer
        self._data = data
        self._answer = answer
        self._model = None
        self.class_names = None
        self._counter = 0
        self.colors = None
        self._prefix = ""
        if len(data)!=1 or len(answer)!=1:
            raise Exception("Data size should be one,",data.shape,answer.shape)
    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter
    @property
    def prefix(self):
        return self._prefix
    @prefix.setter
    def prefix(self,value):
        if value is not None and len(value)>0:
            value+="_"
        else:
            value=""
        self._prefix = value
            
    def on_work_begin(self,worker,**kwargs):
        self._model = worker.model
        self._executor = worker.executor
        self._writer.add_figure("answer_figure",self._answer[0],
                                prefix=self._prefix,colors=self.colors,
                                labels=self.class_names,title="Answer figure",
                                use_stack=True)

    def on_epoch_end(self,**kwargs):
        #Value's shape should be (1,C,L)
        result = self._executor.predict(self._model,self._data,[self._data.shape[2]])[0]
        if hasattr(self._model,'saved_distribution'):
            for name,value in self._model.saved_distribution.items():
                self._writer.add_distribution(name,value,prefix=self._prefix,
                                              counter=self._counter)
                if len(value.shape)==3:
                    value = value[0]
                self._writer.add_figure(name+"_figure",value.transpose(0,1),prefix=self._prefix,
                                        counter=self._counter,title=name)
        class_num = result.shape[0]
        index = result.max(0)[1].cpu().numpy()
        result = index2onehot(index,class_num)
        result = np.transpose(result)
        self._writer.add_figure("result_figure",result,
                                prefix=self._prefix,colors=self.colors,
                                labels=self.class_names,title="Result figure",
                                use_stack=True)
        if hasattr(self._model,'use_CRF'):
            if self._model.use_CRF:
                transitions = self._model.CRF.transitions.cpu().detach().numpy()
                self._writer.add_matshow("transitions_figure",transitions,
                                         prefix=self._prefix,
                                         title="Transitions figure (to row index from column index)")

class DataCallback(Callback):
    def __init__(self):
        self._data = None
        self._prefix = ""
        self._reset()
    @property
    def prefix(self):
        return self._prefix
    @prefix.setter
    def prefix(self,value):
        if value is not None and len(value)>0:
            value+="_"
        else:
            value=""
        self._prefix = value
    @abstractproperty
    def data(self):
        pass
    @abstractmethod
    def _reset(self):
        pass
    def on_work_begin(self,**kwargs):
        self._reset()

class Recorder(DataCallback):
    def __init__(self):
        super().__init__()
        self.path = None
    def _reset(self):
        self._data = {}
    def on_epoch_end(self,metric,**kwargs):
        for type_,value in metric.items():
            if type_ not in self._data.keys():
                self._data[type_] = []
            self._data[type_].append(value)
    def on_work_end(self):
        if self.path is not None:
            df = pd.DataFrame.from_dict(self._data)
            df.to_csv(self.path)
    @property
    def data(self):
        return self._data

class Accumulator(DataCallback):
    def __init__(self):
        super().__init__()
        self.name = ""
        self._batch_count = 0
        self.round_value = 3

    def _reset(self):
        self._data = 0
        self._batch_count = 0

    def on_epoch_begin(self,**kwargs):
        self._reset()

    def on_batch_end(self,metric,**kwargs):
        with torch.no_grad():
            self._data += metric
            self._batch_count += 1
    @property
    def data(self):
        if self._batch_count > 0:
            value = round(self._data/self._batch_count,self.round_value)
            return {self._prefix+self.name:value}
        else:
            return None

class CategoricalMetric(DataCallback):
    def __init__(self):
        self.class_num = 2
        self.ignore_value = -1
        self.show_precision = False
        self.show_recall = False
        self.show_f1 = True
        self.show_acc = True
        self._class_names = None
        self._result = None
        self.round_value = 3
        super().__init__()
    @property
    def class_names(self):
        return self._class_names
    @class_names.setter
    def class_names(self,names):
        if len(names)==self.class_num:
            if names[0]=='exon' and names[1]=='intron' and names[2]=='other':
                self._class_names = list(names)
            else:
                raise Exception('Currently supported tpye should be exon,intron,other')
        else:
            raise Exception('The number of class\'s name is not the same with class\' number')
        
    def on_batch_begin(self,**kwargs):
        self._reset()
    def on_batch_end(self,outputs,labels,**kwargs):
        #N,C,L
        if len(outputs.shape) != 3 or len(labels.shape) != 3:
            raise Exception("Wrong input shape",outputs.shape,labels.shape)
        if outputs.shape[0] != labels.shape[0] or outputs.shape[1] != labels.shape[1]:
            raise Exception("Inconsist batch size or channel size",outputs.shape,labels.shape)
        with torch.no_grad():
            max_length = outputs.shape[2]
            outputs = outputs.max(1)[1].contiguous().view(-1)
            labels = labels.transpose(0,2)[:max_length].transpose(0,2)
            if self.ignore_value is not None:
                mask = (labels.max(1)[0] != self.ignore_value).view(-1)
            else:
                mask = torch.ones(len(outputs))
            labels = labels.max(1)[1].contiguous().view(-1)    
            T_ = (outputs == labels)
            F_ = (outputs != labels)
            T_ = T_*mask
            F_ = F_*mask
            self._data['T'] += T_.sum().item()
            self._data['F'] += F_.sum().item()
            for index in range(self.class_num):
                P = outputs == index
                R = labels == index
                TP_ = P & R
                TN_ = ~P & ~R
                FP_ = P & ~R
                FN_ = ~P & R
                TP_ = (TP_*mask)
                TN_ = (TN_*mask)
                FP_ = (FP_*mask)
                FN_ = (FN_*mask)
                TP = TP_.sum().item()
                TN = TN_.sum().item()
                FP = FP_.sum().item()
                FN = FN_.sum().item()
                self._data["TP_"+str(index)] += TP
                self._data["FP_"+str(index)] += FP
                self._data["TN_"+str(index)] += TN
                self._data["FN_"+str(index)] += FN
            self._result = self._calculate()

    def _calculate(self):
        data = {}
        macro_precision_sum = 0
        for index,val in enumerate(self.precision):
            macro_precision_sum += val
            if self.show_precision:
                if self._class_names is not None:
                    postfix = self.class_names[index]
                else:
                    postfix = str(index)
                data[self.prefix+"precision_"+postfix] = round(val,self.round_value)
        macro_precision = macro_precision_sum/self.class_num
        if self.show_precision:
            data[self._prefix+"macro_precision"] = round(macro_precision,self.round_value)
        macro_recall_sum = 0
        for index,val in enumerate(self.recall):
            macro_recall_sum += val
            if self.show_recall:
                if self.class_names is not None:
                    postfix = self.class_names[index]
                else:
                    postfix = str(index)
                data[self._prefix+"recall_"+postfix] = round(val,self.round_value)
        macro_recall = macro_recall_sum/self.class_num
        if self.show_recall:
            data[self.prefix+"macro_recall"] = round(macro_recall,self.round_value)
        if self.show_acc:
            data[self.prefix+"accuracy"] = round(self.accuracy,self.round_value)
        if self.show_f1:
            for index,val in enumerate(self.F1):
                if self._class_names is not None:
                    postfix = self.class_names[index]
                else:
                    postfix = str(index)
                data[self.prefix+"F1_"+postfix] = round(val,self.round_value)
            if macro_precision+macro_recall > 0:
                macro_F1 = (2*macro_precision*macro_recall)/(macro_precision+macro_recall)
            else:
                macro_F1 = 0
            data[self.prefix+"macro_F1"] = round(macro_F1,self.round_value)
        return data

    @property
    def data(self):
        return self._result
    def _reset(self):
        self._data = {}
        for type_ in ['TP','FP','TN','FN']:
            for index in range(self.class_num):
                self._data[type_+"_"+str(index)] = 0
        self._data['T'] = 0
        self._data['F'] = 0
    @property
    def accuracy(self):
        T = self._data['T']
        F = self._data['F']
        acc = T/(T+F)
        return acc
    @property
    def precision(self):
        precisions = []
        for index in range(self.class_num):
            TP = self._data["TP_"+str(index)]
            FP = self._data["FP_"+str(index)]
            P = (TP+FP)
            if P!=0:
                precision = TP/P
            else:
                precision = 0
            precisions.append(precision)
        return precisions
    @property
    def recall(self):
        recalls = []
        for index in range(self.class_num):
            TP = self._data["TP_"+str(index)]
            FN = self._data["FN_"+str(index)]
            RP = (TP+FN)
            if RP!=0:
                recall = TP/RP
            else:
                recall = 0
            recalls.append(recall)
        return recalls
    @property
    def F1(self):
        f1s = []
        for index in range(self.class_num):
            TP = self._data["TP_"+str(index)]
            FP = self._data["FP_"+str(index)]
            FN = self._data["FN_"+str(index)]
            denominator = (2*TP+FP+FN)
            if denominator!=0:
                f1 = 2*TP/denominator
            else:
                f1 = 0
            f1s.append(f1)
        return f1s