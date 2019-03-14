from abc import ABCMeta, abstractmethod, abstractproperty
import torch.nn.functional as F
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

    def add_scalar(self,record,prefix,counter=None):
        for name,val in record.items():
            self._writer.add_scalar(prefix+name, val, counter)

    def add_grad(self,named_parameters,prefix,counter=None):
        counter = counter or self.counter
        for name,param in named_parameters:
            if param.grad is not None:
                grad = param.grad.cpu().detach().numpy()
                if  np.isnan(grad).any():
                    print(grad)
                    raise Exception(name+" has at least one NaN in it.")
                self._writer.add_histogram("layer_"+prefix+'grad_'+name, grad, counter)

    def add_distribution(self,name,data,prefix,counter=None):
        counter = counter or self.counter
        values = data.contiguous().view(-1).cpu().detach().numpy()
        if  np.isnan(values).any():
            print(values)
            raise Exception(name+" has at least one NaN in it.")
        self._writer.add_histogram(prefix+name, values,counter)

    def add_weights(self,named_parameters,prefix,counter=None):
        counter = counter or self.counter
        for name,param in named_parameters:
            w = param.cpu().detach().numpy()
            if  np.isnan(w).any():
                print(w)
                raise Exception(name+" has at least one NaN in it.")
            self._writer.add_histogram("layer_"+prefix+name, w, counter)

    def add_figure(self,name,value,prefix,counter=None,title='',labels=None,
                   colors=None,use_stack=False,*args,**kwargs):
        counter = counter or self.counter
        #data shape is L,C
        fig = pyplot.figure(dpi=200)
        if isinstance(value,torch.Tensor):
            value = value.cpu().detach().numpy()
        if  np.isnan(value).any():
            print(value)
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
    def on_work_begin(self,model,**kwargs):
        self._model = model
    def on_work_end(self):
        pass
    def on_epoch_begin(self):
        pass
    def on_epoch_end(self,metric,**kwargs):
        pass
    def on_batch_begin(self):
        pass
    def on_batch_end(self,outputs,labels,metric,**kwargs):
        pass

class GFFCompare(Callback):
    def __init__(self,ann_types,path,simplify_map):
        self.extractor = GeneInfoExtractor()
        self.path = path
        self._simplify_map = simplify_map
        self._ann_types = ann_types
        self._counter = 0
    def on_epoch_begin(self):
        self._counter += 1
        self._outputs = SeqInfoContainer()
        self._answers = SeqInfoContainer()
    def on_batch_end(self,index_outputs,labels,ids,lengths,**kwargs):
        C = len(self._ann_types)
        indice = list(range(C))
        for id_,index, label, length in zip(ids,index_outputs,labels,lengths):
            id_ = "around_"+id_
            onehot = index2onehot(index[:length].cpu().numpy(),C)
            seq = vecs2seq(onehot,id_,'plus',self._ann_types)
            seq_informs = self.extractor.extract_per_seq(seq,self._simplify_map)
            self._outputs.add(seq_informs)
            if self._counter==1:
                label = label.transpose(0,1)[:length].transpose(0,1).cpu().numpy()
                ann = vecs2seq(label,id_,'plus',self._ann_types)
                ann_inorms = self.extractor.extract_per_seq(ann,self._simplify_map)
                self._answers.add(ann_inorms)
    def on_epoch_end(self,**kwargs):
        predict_path = self.path+"/predict_"+str(self._counter)+".gff3"
        answer_path = self.path+"/answers.gff3"
        if self._counter==1:
            self._answers.to_gtf().to_csv(answer_path,index=None,header=None,sep='\t')
        if len(self._outputs)>0:
            self._outputs.to_gtf().to_csv(predict_path,index=None,header=None,sep='\t')
            os.system('~/../home/gffcompare/gffcompare -r '+answer_path+' '+\
                      predict_path+' -o '+self.path+"/gffcompare_"+str(self._counter))



class EarlyStop(Callback):
    def __init__(self,target='val_loss',optimized="min",patient=5):
        self.target = target
        self.optimized = optimized
        self.patient = patient
        self._epoch_counter = 0
        self.best_result = None
        self.best_epoch = 0
    def on_epoch_end(self,metric,worker,**kwargs):
        self._epoch_counter+=1
        target = metric[self.target]
        if self.best_result is None:
            self.best_result = target
            self.best_epoch = self._epoch_counter
        else:
            if self.optimized == 'min':
                if self.best_result > target:
                    self.best_epoch = self._epoch_counter
                    self.best_result = target
            elif self.optimized == 'max':
                if self.best_result < target:
                    self.best_epoch = self._epoch_counter
                    self.best_result = target
            else:
                raise Exception("Wrong optimize type")
        if (self._epoch_counter-self.best_epoch)>=self.patient:
            worker._RUNNING=False
    def on_work_end(self):
        print("Best "+str(self.target)+": "+str(self.best_result))

class TensorboardCallback(Callback):
    def __init__(self,tensorboard_writer,prefix=None):
        self._tensorboard_writer = tensorboard_writer
        self._model = None
        self._epoch_counter = 0
        self.do_add_grad = True
        self.do_add_weights = True
        self.do_add_scalar = True
        if prefix is not None:
            prefix+="_"
        else:
            prefix=""
        self._prefix = prefix
        
    def on_epoch_end(self,metric,**kwargs):
        self._epoch_counter+=1
        if self.do_add_grad:
            self._tensorboard_writer.add_grad(counter=self._epoch_counter,
                                              named_parameters=self._model.named_parameters(),prefix=self._prefix)
        if self.do_add_weights:
            self._tensorboard_writer.add_weights(counter=self._epoch_counter,
                                                 named_parameters=self._model.named_parameters(),prefix=self._prefix)
        if self.do_add_scalar:
            self._tensorboard_writer.add_scalar(counter=self._epoch_counter,record=metric,prefix=self._prefix)

class SeqFigCallback(Callback):
    def __init__(self,tensorboard_writer,data,answer,class_names=None,
                 colors=None,train=False,prefix=None):
        self._writer = tensorboard_writer
        self._train = train
        self._data = data
        self._answer = answer
        self._model = None
        self._class_names = class_names
        self._epoch_counter = 0
        self._colors = colors
        if prefix is not None:
            self._prefix = prefix+"_"
        else:
            self._prefix = ""
            
    def on_work_begin(self,model,**kwargs):
        super().on_work_begin(model,**kwargs)
        self._writer.add_figure("answer_figure",self._answer,
                                prefix=self._prefix,colors=self._colors,
                                labels=self._class_names,title="Answer figure",
                                use_stack=True)
    def on_epoch_end(self,**kwargs):
        #Value's shape should be (1,C,L)
        self._epoch_counter += 1 
        self._model.train(self._train)
        lengths = [self._data.shape[2]]
        self._model(self._data,lengths=lengths,return_values=True)
        for name,value in self._model.saved_distribution.items():
            self._writer.add_distribution(name,value,prefix=self._prefix,
                                          counter=self._epoch_counter)
            if len(value.shape)==3:
                value = value[0]
            self._writer.add_figure(name+"_figure",value.transpose(0,1),prefix=self._prefix,
                                    counter=self._epoch_counter,title=name)
        with torch.no_grad():    
            result = self._model.saved_index_outputs[0].cpu().numpy()
        class_num = self._model.out_channels
        result = np.array([(result==index).astype('int') for index in range(class_num)])
        result = np.transpose(result)
        self._writer.add_figure("result_figure",result,
                                prefix=self._prefix,colors=self._colors,
                                labels=self._class_names,title="Result figure",
                                use_stack=True)
        transitions = self._model.CRF.transitions.cpu().detach().numpy()
        print(transitions)
        self._writer.add_matshow("transitions_figure",transitions,
                                 prefix=self._prefix,title="Transitions figure (to row index from column index)")
class DataCallback(Callback):
    def __init__(self,prefix=None):
        self._data = None
        self._model = None
        if prefix is not None and len(prefix)>0:
            prefix+="_"
        else:
            prefix=""
        self._prefix = prefix
        self._reset()
    @abstractproperty
    def data(self):
        pass
    @abstractmethod
    def _reset(self):
        pass

class Recorder(DataCallback):
    def _reset(self):
        self._data = {}
    def on_work_begin(self,**kwargs):
        super().on_work_begin(**kwargs)
        self._reset()
    def on_epoch_end(self,metric,**kwargs):
        for type_,value in metric.items():
            if type_ not in self._data.keys():
                self._data[type_] = []
            self._data[type_].append(value)
    @property
    def data(self):
        return self._data
           
class Accumulator(DataCallback):
    def __init__(self,prefix=None,name=None):
        self._name = name or ""
        self._batch_count = 0
        super().__init__(prefix=prefix)
    def _reset(self):
        self._data = 0
        self._batch_count = 0
    def on_epoch_begin(self):
        self._reset()
    def on_batch_end(self,metric,**kwargs):
        with torch.no_grad():
            self._data+=metric
            self._batch_count+=1
    @property
    def data(self):
        if self._batch_count>0:
            return {self._prefix+self._name:self._data/self._batch_count}
        else:
            return None

class CategoricalMetric(DataCallback):
    def __init__(self, prefix=None,class_num=2, ignore_index=None,
                 class_names=None,show_precision=False,show_recall=False,
                 show_f1=True,show_acc=True,BatchCRF=None):
        self._class_num = class_num
        self._ignore_index = ignore_index
        self._show_precision = show_precision
        self._show_recall = show_recall
        self._show_f1 = show_f1
        self._show_acc = show_acc
        self._BatchCRF = BatchCRF
        if len(class_names)==class_num:
            self._class_names = class_names
        else:
            raise Exception('The number of class\'s name is not the same with class\' number')
        super().__init__(prefix=prefix)
    def on_batch_begin(self):
        self._reset()
    def on_batch_end(self,index_outputs,labels,**kwargs):
        #N,L or N,C,L
        with torch.no_grad():
            max_length = index_outputs.shape[1]
            index_outputs = index_outputs.contiguous().view(-1)
            mask = torch.ones(labels.shape[2])
            if len(labels.shape)==3:
                labels = labels.transpose(0,2)[:max_length].transpose(0,2)
                max_label, labels = labels.max(1)
                labels=labels.view(-1)
                if self._ignore_index is not None:
                    mask = max_label.view(-1) != self._ignore_index
            else:
                labels = labels.transpose(0,1)[:max_length].transpose(0,1)
                labels=labels.view(-1)
                if self._ignore_index is not None:
                    mask = labels != self._ignore_index
            T_ = (index_outputs == labels)
            F_ = (index_outputs != labels)
            T_ = T_*mask
            F_ = F_*mask
            self._data['T'] += T_.sum().item()
            self._data['F'] += F_.sum().item()
            for index in range(self._class_num):
                P = index_outputs == index
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
    @property
    def data(self):
        data = {}
        macro_precision_sum = 0
        for index,val in enumerate(self.precision):
            macro_precision_sum += val
            if self._show_precision:
                if self._class_names is not None:
                    postfix = self._class_names[index]
                else:
                    postfix = str(index)
                data[self._prefix+"precision_"+postfix] = val
        macro_precision = macro_precision_sum/self._class_num
        if self._show_precision:
            data[self._prefix+"macro_precision"] = macro_precision
        macro_recall_sum = 0
        for index,val in enumerate(self.recall):
            macro_recall_sum += val
            if self._show_recall:
                if self._class_names is not None:
                    postfix = self._class_names[index]
                else:
                    postfix = str(index)
                data[self._prefix+"recall_"+postfix] = val
        macro_recall = macro_recall_sum/self._class_num
        if self._show_recall:
            data[self._prefix+"macro_recall"] = macro_recall
        if self._show_acc:
            data[self._prefix+"accuracy"] = self.accuracy
        if self._show_f1:
            for index,val in enumerate(self.F1):
                if self._class_names is not None:
                    postfix = self._class_names[index]
                else:
                    postfix = str(index)
                data[self._prefix+"F1_"+postfix] = val
            if macro_precision+macro_recall > 0:
                macro_F1 = (2*macro_precision*macro_recall)/(macro_precision+macro_recall)
            else:
                macro_F1 = 0
            data[self._prefix+"macro_F1"] = macro_F1
        return data
    def _reset(self):
        self._data = {}
        for type_ in ['TP','FP','TN','FN']:
            for index in range(self._class_num):
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
        for index in range(self._class_num):
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
        for index in range(self._class_num):
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
        for index in range(self._class_num):
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