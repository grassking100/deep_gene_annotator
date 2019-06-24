from abc import ABCMeta, abstractmethod, abstractproperty
import pandas as pd
import torch
from  matplotlib import pyplot
import numpy as np
import os
from ..genome_handler.region_extractor import GeneInfoExtractor
from ..genome_handler.seq_container import SeqInfoContainer
from ..genome_handler.ann_seq_processor import vecs2seq,index2onehot
from .tensorboard_writer import TensorboardWriter
from .metric import F1,accuracy,precision,recall,categorical_metric

class ICallback(metaclass=ABCMeta):
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
    
class Callback(ICallback):
    def __init__(self,prefix)
        self._prefix = ""
        if prefix is None:
            self.prefix = prefix
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
    
class Callbacks(ICallback):
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
    def __init__(self,ann_types,path,simplify_map,prefix=None):
        super().__init__(prefix)
        self.extractor = GeneInfoExtractor()
        self._path = path
        self._simplify_map = simplify_map
        self._ann_types = ann_types
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
    def __init__(self,target=None,optimize_min=None,patient=None,path=None,period=None,
                 save_best_weights=None,restore_best_weights=None,prefix=prefix):
        super().__init__(prefix)
        self.target = target or 'val_loss'
        self.optimize_min = optimize_min or True
        self.patient = 16
        self.path = path
        self.period = period
        self.save_best_weights = save_best_weights or False
        self.restore_best_weights = restore_best_weights or False
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
    def __init__(self,tensorboard_writer,prefix=None):
        super().__init__(prefix)
        self.tensorboard_writer = tensorboard_writer
        self._model = None
        self._counter = 0
        self.do_add_grad = False
        self.do_add_weights = False
        self.do_add_scalar = True

    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter
        
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
    def __init__(self,tensorboard_writer,data,answer,prefix=None):
        super().__init__(prefix)
        self._writer = tensorboard_writer
        self._data = data
        self._answer = answer
        self._model = None
        self.class_names = None
        self._counter = 0
        self.colors = None
        if len(data)!=1 or len(answer)!=1:
            raise Exception("Data size should be one,",data.shape,answer.shape)

    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter
            
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
    def __init__(self,prefix=None):
        super().__init__(prefix)
        self._data = None
        self._reset()
    @abstractproperty
    def data(self):
        pass
    @abstractmethod
    def _reset(self):
        pass
    def on_work_begin(self,**kwargs):
        self._reset()

class Recorder(DataCallback):
    def __init__(self,prefix=None):
        super().__init__(prefix)
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
    def __init__(self,name=None,prefix=None):
        super().__init__(prefix)
        self.name = name or ""
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
    def __init__(self,prefix=None):
        super().__init__(prefix)
        self.class_num = 2
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

    def on_batch_end(self,outputs,labels,mask=None,**kwargs):
        #N,C,L
        data = categorical_metric(outputs,labels,mask)
        for key in data.keys():
            self._data[key] = data[key]
        self._result = self._calculate()

    def _calculate(self,T,F,TPs,FPs,TNs,FNs):
        recall_ = recall(TPs,FNs)
        precision_ = precision(TPs,FPs)
        accuracy_ = accuracy(T,F)
        f1 = F1(TPs,FPs,FNs)
        data = {}
        macro_precision_sum = 0
        for index,val in enumerate(precision_):
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
        for index,val in enumerate(recall_):
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
            data[self.prefix+"accuracy"] = round(accuracy_,self.round_value)
        if self.show_f1:
            for index,val in enumerate(f1):
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
            self._data[type_] = [0]*self.class_num
        self._data['T'] = 0
        self._data['F'] = 0
