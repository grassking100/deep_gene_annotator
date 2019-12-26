import os
import warnings
from abc import abstractmethod, abstractproperty, ABCMeta
import deepdish as dd
import json
import math
import numpy as np
import pandas as pd
import torch
import seqlogo
from skimage.filters import threshold_otsu
from ..utils.utils import create_folder,write_gff,gff_to_bed_command,gffcompare_command,save_as_gff_and_bed
from ..utils.seq_converter import DNA_CODES
from ..genome_handler.seq_container import EmptyContainerException
from .metric import calculate_metric,categorical_metric,contagion_matrix
from .warning import WorkerProtectedWarning
from .inference import ann_vec2one_hot_vec,basic_inference
from .signal import get_signal_ppm, ppms2meme

class ICallback(metaclass=ABCMeta):
    @abstractmethod
    def get_config(self,**kwargs):
        pass
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
    def __init__(self,prefix=None,*args,**kwargs):
        self._prefix = ""
        self.prefix = prefix

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self,prefix):
        if prefix is not None and len(prefix)>0:
            prefix+="_"
        else:
            prefix=""
        self._prefix = prefix

    def get_config(self,**kwargs):
        config = {}
        config['prefix'] = self._prefix
        return config

class Callbacks(ICallback):
    def __init__(self,callbacks=None):
        self._callbacks = []
        if callbacks is not None:
            self.add(callbacks)

    def get_config(self,**kwargs):
        config = {}
        for callback in self._callbacks:
            name = callback.__class__.__name__
            config[name] = callback.get_config()
        return config

    def clean(self):
        self._callbacks = []

    @property
    def callbacks(self):
        return self._callbacks

    def add(self,callbacks):
        list_ = []
        if isinstance(callbacks,Callbacks):
            for callback in callbacks.callbacks:
                if isinstance(callback,Callbacks):
                    list_ += callback.callbacks
                else:
                    list_ += [callback]
        elif isinstance(callbacks,list):
            list_ += callbacks
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
    def __init__(self,ann_vec2info_converter,path,fix_boundary=False,prefix=None):
        super().__init__(prefix)
        if ann_vec2info_converter is None:
            raise Exception("The ann_vec2info_converter should not be None")
        if path is None:
            raise Exception("The path should not be None")
        self.converter = ann_vec2info_converter
        self.fix_boundary = fix_boundary
        self._path = path
        self._counter = None
        self._outputs = None
        self._answers = None
        self._model = None
        
    def on_work_begin(self,worker,**kwargs):
        self._counter = 0
        self._model = worker.model

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self._path
        config['ann_vec2info_converter'] = self.converter.get_config()
        config['fix_boundary'] = self.fix_boundary
        return config

    def on_epoch_begin(self,counter,**kwargs):
        self._outputs = []
        self._answers = []
        self._counter = counter

    def on_batch_end(self,predict_result,seq_data,masks,**kwargs):
        chrom_ids,dna_seqs,answers,lengths = seq_data.ids,seq_data.seqs,seq_data.answers,seq_data.lengths
        answers = answers.cpu().numpy()
        predict_result = predict_result.cpu().numpy()
        try:
            if self.fix_boundary:
                output_info = self.converter.vecs2fixed_info(chrom_ids,lengths,dna_seqs,predict_result)
            else:
                output_info = self.converter.vecs2info(chrom_ids,lengths,predict_result)
            self._outputs.append(output_info)
        except EmptyContainerException:
            pass

        if self._counter == 1:
            try:
                label_info = self.converter.vecs2info(chrom_ids,lengths,answers)
                self._answers.append(label_info)
            except EmptyContainerException:
                pass

    def on_epoch_end(self,**kwargs):
        prefix = "{}gffcompare_{}".format(self.prefix,self._counter)
        prefix_path = os.path.join(self._path,prefix)
        answer_bed_path = os.path.join(self._path,"answers.bed")
        answer_gff_path = os.path.join(self._path,"answers.gff3")
        predict_bed_path = os.path.join(self._path,"{}.bed".format(prefix))
        predict_gff_path = os.path.join(self._path,"{}.gff3".format(prefix))
        if self._counter == 1:
            answers = pd.concat(self._answers)
            save_as_gff_and_bed(answers,answer_gff_path,answer_bed_path)

        if len(self._outputs) > 0:
            outputs = pd.concat(self._outputs)
            save_as_gff_and_bed(outputs,predict_gff_path,predict_bed_path)
            gffcompare_command(answer_gff_path,predict_gff_path,prefix_path)
            
class SignalSaver(Callback):
    def __init__(self,path,prefix=None):
        super().__init__(prefix)
        if path is None:
            raise Exception("The path should not be None")
        self._path = path
        self._raw_outputs = None
        self._raw_answers = None
        
    def on_work_begin(self,worker,**kwargs):
        self._counter = 0
        self._model = worker.model

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self._path
        return config

    def on_epoch_begin(self,counter,**kwargs):
        self._raw_outputs = []
        self._raw_answers = []
        self._counter = counter

    def on_batch_end(self,outputs,seq_data,**kwargs):
        chrom_ids,dna_seqs,answers,lengths = seq_data.ids,seq_data.seqs,seq_data.answers,seq_data.lengths
        answers = answers.cpu().numpy()
        self._raw_outputs.append({'outputs':outputs,
                                  'chrom_ids':chrom_ids,
                                  'dna_seqs':dna_seqs,
                                  'lengths':lengths})

        if self._counter == 1:
            self._raw_answers.append({'answers':answers,
                                      'chrom_ids':chrom_ids,
                                      'dna_seqs':dna_seqs,
                                      'lengths':lengths})

    def on_epoch_end(self,**kwargs):
        raw_output_path = os.path.join(self._path,'{}raw_output_{}.h5').format(self.prefix,self._counter)
        raw_answer_path = os.path.join(self._path,'{}raw_answer.h5').format(self.prefix)
        dd.io.save(raw_output_path , self._raw_outputs)

        if self._counter == 1:
            dd.io.save(raw_answer_path, self._raw_answers)
        
class TensorboardCallback(Callback):
    def __init__(self,tensorboard_writer,prefix=None):
        super().__init__(prefix)
        self.tensorboard_writer = tensorboard_writer
        self._model = None
        self._counter = None
        self.do_add_grad = False
        self.do_add_weights = False
        self.do_add_scalar = True

    def on_work_begin(self,worker,**kwargs):
        self._model = worker.model
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['do_add_grad'] = self.do_add_grad
        config['do_add_weights'] = self.do_add_weights
        config['do_add_scalar'] = self.do_add_scalar
        return config

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
        
    def on_work_end(self):
        self.tensorboard_writer.close()

class SeqFigCallback(Callback):
    def __init__(self,tensorboard_writer,data,answer,label_names=None,colors=None,prefix=None):
        super().__init__(prefix)
        self._writer = tensorboard_writer
        self._data = data
        self._answer = answer
        self._model = None
        self._counter = None
        self.label_names = label_names
        self.colors = colors
        self.do_add_distribution = True
        self._executor = None
        if len(data)!=1 or len(answer)!=1:
            raise Exception("Data size should be one,",data.shape,answer.shape)

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['label_names'] = self.label_names
        config['colors'] = self.colors
        return config

    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter

    def on_work_begin(self,worker,**kwargs):
        self._model = worker.model
        self._executor = worker.executor
        self._writer.add_figure("answer_figure",self._answer[0],
                                prefix=self._prefix,colors=self.colors,
                                labels=self.label_names,title="Answer figure",
                                use_stack=True)

    def on_epoch_end(self,**kwargs):
        #Value's shape should be (1,C,L)
        predict_result = self._executor.predict(self._model,self._data,[self._data.shape[2]])[0]
        if self.do_add_distribution and hasattr(self._model,'saved_distribution'):
            for name,value in self._model.saved_distribution.items():
                self._writer.add_distribution(name,value,prefix=self._prefix,
                                              counter=self._counter)
                if len(value.shape)==3:
                    value = value[0]
                value = value.transpose()
                self._writer.add_figure(name+"_figure",value,prefix=self._prefix,
                                        counter=self._counter,title=name)
        predict_result = predict_result.cpu().numpy()[0]
        C,L = predict_result.shape
        onehot = ann_vec2one_hot_vec(predict_result)
        onehot = np.transpose(onehot)
        self._writer.add_figure("result_figure",onehot,prefix=self._prefix,colors=self.colors,
                                labels=self.label_names,title="Result figure",use_stack=True)
        diff = np.transpose(predict_result) - self._answer[0][:L,:]
        self._writer.add_figure("diff_figure",diff,prefix=self._prefix,colors=self.colors,
                                labels=self.label_names,title="Predict - Answer figure",use_stack=False)
        
class DataCallback(Callback):
    def __init__(self,prefix=None):
        super().__init__(prefix)
        self._data = None

    @abstractproperty
    def data(self):
        pass

    @abstractmethod
    def _reset(self):
        pass

    def on_work_begin(self,**kwargs):
        self._reset()

class Recorder(DataCallback):
    def __init__(self,prefix=None,path=None,force_reset=False):
        super().__init__(prefix)
        self.path = path
        self._force_reset = force_reset
        self._worker = None

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self.path
        config['force_reset'] = self._force_reset
        return config

    def _reset(self):
        self._data = {}
        if self.path is not None and os.path.exists(self.path) and not self._force_reset:
            with open(self.path,'r') as fp:
                self._data = json.load(fp)

    def on_work_begin(self,epoch_start=None,**kwargs):
        super().on_work_begin()
        if epoch_start is not None:
            for type_,value in self._data.items():
                self._data[type_] = value[:epoch_start-1]

    def on_epoch_end(self,metric,**kwargs):
        for type_,value in metric.items():
            if type_ not in self._data.keys():
                self._data[type_] = []
            self._data[type_].append(value)

        if self.path is not None:
            with open(self.path,'w') as fp:
                json.dump(self.data,fp, indent=4)

    @property
    def data(self):
        return self._data

class Accumulator(DataCallback):
    def __init__(self,prefix=None):
        super().__init__(prefix)
        self._batch_count = None
        self._epoch_count = None
        self.round_value = 3
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['round_value'] = self.round_value
        return config

    def _reset(self):
        self._data = {}
        self._batch_count = 0
        self._epoch_count = 0

    def on_epoch_begin(self,**kwargs):
        self._reset()
        self._epoch_count += 1

    def on_batch_end(self,metric,**kwargs):
        if self._batch_count == 0:
            for key in metric.keys():
                self._data[key] = 0
        for key,value in metric.items():
            self._data[key] += value
        self._batch_count += 1

    @property
    def data(self):
        if self._batch_count > 0:
            data = {}
            for key,value in self._data.items():
                value = round(value/self._batch_count,self.round_value)
                data[self._prefix+key] = value
            return data
        else:
            return None

class CategoricalMetric(DataCallback):
    def __init__(self,label_num,label_names=None,prefix=None):
        super().__init__(prefix)
        self.label_num = label_num or 3
        self.answer_inference = basic_inference(self.label_num)
        self.show_precision = False
        self.show_recall = False
        self.show_f1 = True
        self.show_acc = True
        self._label_names = None
        if label_names is not None:
            self.label_names = label_names
        self._result = None
        self.round_value = 3

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['label_num'] = self.label_num
        config['show_precision'] = self.show_precision
        config['show_recall'] = self.show_recall
        config['show_f1'] = self.show_f1
        config['show_acc'] = self.show_acc
        config['label_names'] = self._label_names
        config['round_value'] = self.round_value
        return config

    @property
    def label_names(self):
        return self._label_names

    @label_names.setter
    def label_names(self,names):
        if len(names)!=self.label_num:
            raise Exception('The number of class\'s name is not the same with class\' number')
        self._label_names = names

    def on_epoch_begin(self,**kwargs):
        self._reset()

    def on_batch_end(self,predict_result,seq_data,masks,**kwargs):
        labels = self.answer_inference(seq_data.answers,masks)
        #N,C,L
        data = categorical_metric(predict_result.cpu().numpy(),
                                  labels.cpu().numpy(),
                                  masks.cpu().numpy())
        for type_ in ['TPs','FPs','TNs','FNs']:
            for index in range(self.label_num):
                self._data[type_][index] += data[type_][index]
        self._data['T'] += data['T']
        self._data['F'] += data['F']
        self._result = calculate_metric(self._data,prefix=self.prefix,
                                        label_names=self.label_names,
                                        calculate_precision=self.show_precision,
                                        calculate_recall=self.show_recall,
                                        calculate_F1=self.show_f1,
                                        calculate_accuracy=self.show_acc,
                                        round_value=self.round_value)

    @property
    def data(self):
        return self._result

    def _reset(self):
        self._data = {}
        for type_ in ['TPs','FPs','TNs','FNs']:
            self._data[type_] = [0]*self.label_num
        self._data['T'] = 0
        self._data['F'] = 0

class ContagionMatrix(DataCallback):
    def __init__(self,label_num,label_names=None,prefix=None):
        super().__init__(prefix)
        self.label_num = label_num or 3
        self.inference = basic_inference(self.label_num)
        self._label_names = None
        if label_names is not None:
            self.label_names = label_names

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['label_num'] = self.label_num
        config['label_names'] = self._label_names
        return config

    @property
    def label_names(self):
        return self._label_names

    @label_names.setter
    def label_names(self,names):
        if len(names)!=self.label_num:
            raise Exception('The number of class\'s name is not the same with class\' number')
        self._label_names = names

    def on_epoch_begin(self,**kwargs):
        self._reset()

    def on_batch_end(self,predict_result,seq_data,masks,**kwargs):
        labels = self.inference(seq_data.answers,masks)
        #N,C,L
        data = contagion_matrix(predict_result.cpu().numpy(),
                                labels.cpu().numpy(),
                                masks.cpu().numpy())
        self._data += np.array(data)

    @property
    def data(self):
        return {self.prefix+'contagion_matrix':self._data.tolist()}

    def _reset(self):
        self._data = np.array([[0]*self.label_num] * self.label_num)
        
class SeqLogo(Callback):
    def __init__(self,saved_distribution_name,saved_root,ratio=None,radius=None,prefix=None):
        super().__init__(prefix)
        self.saved_root = saved_root
        self.saved_distribution_name = saved_distribution_name
        self.ratio = ratio
        self.radius = radius
        self._lengths = None
        self._seqs = None
        self._model = None
        self._png_root = None
        self._ppm_root = None
        self._signals = None
        
    def get_config(self,**kwargs):
        config = super().get_config()
        config['saved_root'] = self.saved_root
        config['saved_distribution_name'] = self.saved_distribution_name
        config['ratio'] = self.ratio
        config['radius'] = self.radius
        return config
        
    def on_work_begin(self,worker,**kwargs):
        self._lengths = []
        self._seqs = []
        logo_root = os.path.join(self.saved_root,'logo')
        self._plot_root = os.path.join(logo_root,'plot')
        self._ppm_root = os.path.join(logo_root,'ppm')
        self._meme_path = os.path.join(logo_root,'motif.meme')
        create_folder(self.saved_root)
        create_folder(logo_root)
        create_folder(self._plot_root)
        create_folder(self._ppm_root)
        self._model = worker.model
        self._signals = {}

    def on_batch_end(self,seq_data,**kwargs):
        lengths,inputs = seq_data.lengths,seq_data.inputs
        self._lengths += lengths.tolist()
        self._seqs += inputs.tolist()
        for name,val in self._model.saved_distribution.items():
            if self.saved_distribution_name in name:
                signals = self._model.saved_distribution[name].cpu().detach().numpy()
                signals = np.split(signals,signals.shape[1],axis=1)
                for index,signal in enumerate(signals):
                    signal_name = "{}_{}".format(name,index)
                    if signal_name not in self._signals:
                        self._signals[signal_name] = []
                    self._signals[signal_name] += np.split(signal,len(signal))

    def on_epoch_end(self,**kwargs):
        sum_ = None
        for seq,length in zip(self._seqs,self._lengths):
            count = np.array(seq)[:length].sum(1)
            if sum_ is None:
                sum_ = count
            else:
                sum_ += count
        background_freq = sum_/sum_.sum()
        plot_file_root = os.path.join(self._plot_root,'{}.png')
        ppm_file_root = os.path.join(self._ppm_root,'{}.csv')
        names = []
        ppm_dfs = []
        seqs = [np.array(seq) for seq in self._seqs]
        for name,signal in self._signals.items():
            #Get ppm
            ppm = get_signal_ppm(signal,seqs,self._lengths,
                                 ratio=self.ratio,radius=self.radius)
            if ppm is not None:
                #Get valid location of ppm
                ppm_df = pd.DataFrame(ppm.T,columns =list(DNA_CODES))
                seqlogo_ppm = seqlogo.Ppm(ppm_df,background=background_freq)
                info_sum = seqlogo_ppm.ic
                threshold = threshold_otsu(info_sum, nbins=1000)
                valid_loc = np.where(info_sum>=threshold)[0]
                if len(valid_loc) > 0:
                    #Get valid location of ppm
                    shape = ppm.shape
                    ppm = ppm[:,min(valid_loc):max(valid_loc)+1]
                    ppm_df = pd.DataFrame(ppm.T,columns = list(DNA_CODES))
                    seqlogo_ppm = seqlogo.Ppm(ppm_df,background=background_freq)
                    #Save seqlogo
                    seqlogo.seqlogo(seqlogo_ppm, ic_scale = True, format = 'png', size = 'xlarge',
                                    filename=plot_file_root.format(name),
                                    stacks_per_line=100,number_interval=20)
                    #Save ppm
                    ppm_df.to_csv(ppm_file_root.format(name),index=None)
                    ppm_dfs.append(ppm_df)
                    names.append(name)
        #Save motif as MEME format
        index = [DNA_CODES.index(alpha) for alpha in list("ACGT")]
        background_freq_ = background_freq[index]
        ppms2meme(ppm_dfs,names,self._meme_path,strands='+',background=background_freq_)

class OptunaCallback(Callback):
    def __init__(self,trial,target=None,prefix=None):
        super().__init__(prefix)
        self.trial = trial
        self.target = target or 'val_loss'
        self._counter = None
        self._worker = None
        self.is_prune = None
        
    def on_work_begin(self, worker,**kwargs):
        self._counter = 0
        self._worker = worker
        if not hasattr(self._worker,'best_epoch') or not hasattr(self._worker,'best_result'):
            raise Exception("The worker should has best_epoch and best_result to work with OptunaCallback")
        self.is_prune = False

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['target'] = self.target
        return config

    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter

    def on_epoch_end(self,metric,**kwargs):
        target = metric[self.target]
        if str(target) == 'nan':
            return

        self.trial.report(target,self._counter)
        if self.trial.should_prune():
            self.is_prune = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=WorkerProtectedWarning)
                self._worker.is_running = False
