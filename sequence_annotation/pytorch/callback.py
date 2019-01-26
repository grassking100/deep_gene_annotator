from abc import ABCMeta, abstractmethod, abstractproperty
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def plt_dynamic(fig,ax,x,ys):
    for y in ys:
        ax.plot(x, y)
    fig.canvas.draw()

class Callback(metaclass=ABCMeta):
    def on_work_begin(self,*args):
        pass
    def on_work_end(self,*args):
        pass
    def on_epoch_begin(self,*args):
        pass
    def on_epoch_end(self,*args):
        pass

class Reporter(Callback):
    def __init__(self):
        self._epoch_count = 0
    def on_work_begin(self,*args):
        self.fig,self.ax = plt.subplots(1,1)
    def on_epoch_end(self,record):
        self._epoch_count+=1
        print(self._epoch_count,record)
        
class ModelCallback(Callback):
    def __init__(self):
        self._model = None
        self._reset()
    @property
    def model(self):
        return self._model

class DataCallback(Callback):
    def __init__(self,prefix=None):
        self._data = None
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
    def on_work_begin(self):
        self._reset()
    def on_epoch_end(self,metric):
        for type_,value in metric.items():
            if type_ not in self._data.keys():
                self._data[type_] = []
            self._data[type_].append(value)
    @property
    def data(self):
        return self._data
            
class DataDeepCallback(DataCallback):
    @abstractmethod
    def on_batch_begin(self):
        pass
    @abstractmethod
    def on_batch_end(self,*args):
        pass

class Accumulator(DataDeepCallback):
    def __init__(self,prefix=None,name=None):
        self._name = name or ""
        self._batch_count = 0
        super().__init__(prefix=prefix)
    def _reset(self):
        self._data = 0
        self._batch_count = 0
    def on_epoch_begin(self):
        self._reset()
    def on_batch_begin(self):
        pass
    def on_batch_end(self,metric):
        with torch.no_grad():
            self._data+=metric
            self._batch_count+=1
    @property
    def data(self):
        return {self._prefix+self._name:self._data/self._batch_count}

class CategoricalMetric(DataDeepCallback):
    def __init__(self, prefix=None,class_num=2, ignore_index=None,
                 show_precision=False,show_recall=False,show_f1=True,show_acc=True):
        self._class_num = class_num
        self._ignore_index = ignore_index
        self._show_precision = show_precision
        self._show_recall = show_recall
        self._show_f1 = show_f1
        self._show_acc = show_acc
        super().__init__(prefix=prefix)
    def on_batch_begin(self):
        self._reset()
    @property
    def data(self):
        data = {}
        if self._show_precision:
            for index,val in enumerate(self.precision):
                data[self._prefix+"precision_"+str(index)] = val
        if self._show_recall:
            for index,val in enumerate(self.recall):
                data[self._prefix+"recall_"+str(index)] = val
        if self._show_acc:
            data[self._prefix+"accuracy"] = self.accuracy
        if self._show_f1:
            for index,val in enumerate(self.F1):
                data[self._prefix+"F1_"+str(index)] = val
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
    def on_batch_end(self,outputs,labels):
        with torch.no_grad(): 
            outputs = outputs.contiguous().view(-1, self._class_num)
            outputs = outputs.max(1)[1]
            labels = torch.transpose(labels, 1, 2).contiguous().view(-1, self._class_num) 
            labels,label_index = labels.max(1)
            if self._ignore_index is not None:
                mask = labels != self._ignore_index
            T_ = (outputs == label_index)
            F_ = (outputs != label_index)
            if self._ignore_index is not None:
                T_ = T_*mask
                F_ = F_*mask
            self._data['T'] += T_.sum().item()
            self._data['F'] += F_.sum().item()
            for index in range(self._class_num):
                P = outputs == index
                R = label_index == index
                TP_ = P & R
                TN_ = ~P & ~R
                FP_ = P & ~R
                FN_ = ~P & R
                if self._ignore_index is not None:
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