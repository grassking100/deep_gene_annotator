import os
import warnings
from abc import abstractmethod, abstractproperty, ABCMeta
import json
import math
import numpy as np
import pandas as pd
import torch
import seqlogo
from skimage.filters import threshold_otsu
from ..utils.utils import create_folder
from ..utils.seq_converter import DNA_CODES
from ..genome_handler.region_extractor import GeneInfoExtractor
from ..genome_handler.seq_container import SeqInfoContainer
from .metric import F1,accuracy,precision,recall,categorical_metric,contagion_matrix
from .warning import WorkerProtectedWarning
from .inference import ann_seq2one_hot_seq, AnnSeq2InfoConverter,basic_inference
from .signal import get_signal_ppm, ppms2meme
from .utils import get_copied_state_dict

preprocess_src_root = 'sequence_annotation/sequence_annotation/preprocess'

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
    def __init__(self,prefix=None,**kwargs):
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
    def __init__(self,ann_types,path,simplify_map,dist,prefix=None):
        super().__init__(prefix)
        self.converter = AnnSeq2InfoConverter(ann_types,simplify_map,dist)
        self.answer_inference = basic_inference(len(ann_types))
        self.fix_boundary = False
        self._path = path
        self._counter = None
        
    def on_work_begin(self,**kwargs):
        self._counter = 0

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self._path
        config['simplify_map'] = self.converter.simplify_map
        config['ann_types'] = self.converter.ann_types
        config['dist'] = self.converter.dist
        config['donor_site_pattern'] = self.converter.donor_site_pattern
        config['accept_site_pattern'] = self.converter.accept_site_pattern
        config['converter'] = str(self.converter)
        config['fix_boundary'] = self.fix_boundary
        config['alt'] = self.converter.extractor.alt
        config['alt_num'] = self.converter.extractor.alt_num
        return config

    def on_epoch_begin(self,counter,**kwargs):
        self._outputs = SeqInfoContainer()
        self._answers = SeqInfoContainer()
        self._counter = counter

    def on_batch_end(self,outputs,seq_data,masks,**kwargs):
        chrom_ids,seqs,answers,lengths = seq_data.ids,seq_data.seqs,seq_data.answers,seq_data.lengths
        answers = self.answer_inference(answers,masks).cpu().numpy()
        outputs = outputs.cpu().numpy()
        output_info = self.converter.convert(chrom_ids,seqs,outputs,lengths,fix_boundary=self.fix_boundary)
        self._outputs.add(output_info)
        if self._counter == 1:
            label_info = self.converter.convert(chrom_ids,seqs,answers,lengths,fix_boundary=False)
            self._answers.add(label_info)

    def on_epoch_end(self,**kwargs):
        path = os.path.join(self._path,"{}gffcompare_{}").format(self.prefix,self._counter)
        answer_path = os.path.join(self._path,"answers")
        to_bed_command = 'python3 {}/gff2bed.py -i {}.gff3 -o {}.bed'
        gffcompare_command = 'gffcompare --strict-match --no-merge -T -r {}.gff3 {}.gff3  -o {}'
        if self._counter == 1:
            self._answers.to_gff().to_csv(self._path+"/answers.gff3",index=None,header=None,sep='\t')
            command = to_bed_command.format(preprocess_src_root,answer_path,answer_path)
            os.system(command)

        if not self._outputs.is_empty():
            self._outputs.to_gff().to_csv(path+".gff3",index=None,header=None,sep='\t')
            command = gffcompare_command.format(answer_path,path,path)
            os.system(command)
            os.system('rm {}.annotated.gtf'.format(path))
            os.system('rm {}.loci'.format(path))
            command = to_bed_command.format(preprocess_src_root,path,path)
            os.system(command)

class ModelCheckpoint(Callback):
    def __init__(self,target=None,optimize_min=True,patient=None,path=None,period=None,
                 save_best_weights=False,restore_best_weights=False,prefix=None):
        super().__init__(prefix)
        self.target = target or 'val_loss'
        self.optimize_min = optimize_min
        self.patient = patient or 16
        self.path = path
        self.period = period
        self.save_best_weights = save_best_weights
        self.restore_best_weights = restore_best_weights
        self._counter = None
        self._best_result = None
        self._best_epoch = None
        self._model_weights = None
        self._worker = None
        
    def on_work_begin(self, worker,**kwargs):
        self._counter = 0
        self._best_result = None
        self._best_epoch = 0
        self._model_weights = None
        if self.save_best_weights:
            self._model_weights = get_copied_state_dict(worker.model)
        self._worker = worker

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['target'] = self.target
        config['optimize_min'] = self.optimize_min
        config['patient'] = self.patient
        config['period'] = self.period
        config['save_best_weights'] = self.save_best_weights
        config['restore_best_weights'] = self.restore_best_weights
        return config

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
                model_path = os.path.join(self.path,'model_epoch_{}.pth').format(self._counter)
                print("Save model at "+model_path)
                torch.save(self._worker.model.state_dict(),model_path)

    def on_epoch_end(self,metric,**kwargs):
        target = metric[self.target]
        if str(target) == 'nan':
            return
        update = False
        if self._best_result is None:
            update = True
        else:
            if self.optimize_min:
                if self._best_result > target:
                    update = True
            else:
                if self._best_result < target:
                    update = True

        if update:
            if self.save_best_weights:
                print("Save best weight of epoch {}".format(self._counter))
                self._model_weights = get_copied_state_dict(self._worker.model)
            self._best_epoch = self._counter
            self._best_result = target
            print("Update best weight of epoch {}".format(self._counter))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=WorkerProtectedWarning)
                self._worker.best_epoch = self._best_epoch
        
        if self.patient is not None:
            #if (self._counter-self.best_epoch) > self.patient:
            #to if (self._counter-self.best_epoch) >= self.patient:
            if (self._counter-self.best_epoch) >= self.patient:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore",category=WorkerProtectedWarning)
                    self._worker.is_running = False

    def on_work_end(self):
        print("Best "+str(self.target)+": "+str(self._best_result))
        if self.save_best_weights:
            if self.restore_best_weights:
                print("Restore best weight of epoch {}".format(self._best_epoch))
                self._worker.model.load_state_dict(self._model_weights)
            if self.path is not None:
                model_path =  os.path.join(self.path,'best_model.pth').format(self._best_epoch)
                best_status_path =  os.path.join(self.path,'best_model.status')
                print("Save best model at "+model_path)
                torch.save(self._model_weights,model_path)
                with open(best_status_path,"w") as fp:
                    #Change 'formula':'(self._counter-self.best_epoch) > self.patient'}
                    # to 'formula':'(self._counter-self.best_epoch) >= self.patient'}
                    best_status = {'best_epoch':self.best_epoch,'path':model_path,
                                   'formula':'(self._counter-self.best_epoch) >= self.patient'}
                    json.dump(best_status,fp, indent=4)

        if self.best_epoch == 0 and self.path is not None:
            model_path =  os.path.join(self.path,'last_model.pth')
            torch.save(self._worker.model.state_dict(),model_path)

class TensorboardCallback(Callback):
    def __init__(self,tensorboard_writer,prefix=None):
        super().__init__(prefix)
        self.tensorboard_writer = tensorboard_writer
        self._model = None
        self._counter = None
        self.do_add_grad = True
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
        outputs,lengths,masks = self._executor.predict(self._model,self._data,[self._data.shape[2]])
        if self.do_add_distribution and hasattr(self._model,'saved_distribution'):
            for name,value in self._model.saved_distribution.items():
                self._writer.add_distribution(name,value,prefix=self._prefix,
                                              counter=self._counter)
                if len(value.shape)==3:
                    value = value[0]
                self._writer.add_figure(name+"_figure",value.transpose(0,1),prefix=self._prefix,
                                        counter=self._counter,title=name)
        outputs = outputs.cpu().numpy()[0]
        C,L = outputs.shape
        onehot = ann_seq2one_hot_seq(outputs)
        onehot = np.transpose(onehot)
        self._writer.add_figure("result_figure",onehot,prefix=self._prefix,colors=self.colors,
                                labels=self.label_names,title="Result figure",use_stack=True)
        diff = np.transpose(outputs) - self._answer[0][:L,:]
        self._writer.add_figure("diff_figure",diff,prefix=self._prefix,colors=self.colors,
                                labels=self.label_names,title="Predict - Answer figure",use_stack=False)

        
class WarningRecorder(Callback):
    def __init__(self,prefix=None,path=None):
        super().__init__(prefix)
        self._data = None
        self.path = path

    def on_work_begin(self,**kwargs):
        self._data = []
    
    def on_epoch_end(self,warnings,**kwargs):
        self._data += warnings

    def on_work_end(self):
        if self.path is not None:
            with open(self.path,'w') as fp:
                for warning in self._data:
                    fp.write("{}\n".format(warning))
        
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
    def __init__(self,prefix=None,path=None):
        super().__init__(prefix)
        self.path = path
        self._worker = None

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self.path
        return config

    def _reset(self):
        self._data = {}

    def on_epoch_end(self,metric,**kwargs):
        for type_,value in metric.items():
            if type_ not in self._data.keys():
                self._data[type_] = []
            self._data[type_].append(value)
            
    def on_work_end(self):
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
        with torch.no_grad():
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

    def on_batch_end(self,outputs,seq_data,masks,**kwargs):
        labels = self.answer_inference(seq_data.answers,masks)
        #N,C,L
        data = categorical_metric(outputs,labels,masks)
        for type_ in ['TPs','FPs','TNs','FNs']:
            for index in range(self.label_num):
                self._data[type_][index] += data[type_][index]
        self._data['T'] += data['T']
        self._data['F'] += data['F']
        self._result = self._calculate()

    def _calculate(self):
        T,F = self._data['T'], self._data['F']
        TPs,FPs = self._data['TPs'], self._data['FPs']
        TNs,FNs = self._data['TNs'], self._data['FNs']
        data = {}
        for name,value in zip(['T','F'],[T,F]):
            data[self.prefix+name] = value
        recall_ = recall(TPs,FNs)
        precision_ = precision(TPs,FPs)
        accuracy_ = accuracy(T,F)
        f1 = F1(TPs,FPs,FNs)

        macro_precision_sum = 0
        for index,val in enumerate(precision_):
            macro_precision_sum += val
            if self.show_precision:
                if self.label_names is not None:
                    postfix = self.label_names[index]
                else:
                    postfix = str(index)
                data[self.prefix+"precision_"+postfix] = round(val,self.round_value)
        macro_precision = macro_precision_sum/self.label_num
        if self.show_precision:
            data[self._prefix+"macro_precision"] = round(macro_precision,self.round_value)
        macro_recall_sum = 0
        for index,val in enumerate(recall_):
            macro_recall_sum += val
            if self.show_recall:
                if self.label_names is not None:
                    postfix = self.label_names[index]
                else:
                    postfix = str(index)
                data[self._prefix+"recall_"+postfix] = round(val,self.round_value)
        macro_recall = macro_recall_sum/self.label_num
        if self.show_recall:
            data[self.prefix+"macro_recall"] = round(macro_recall,self.round_value)
        if self.show_acc:
            data[self.prefix+"accuracy"] = round(accuracy_,self.round_value)
        if self.show_f1:
            for index,val in enumerate(f1):
                if self.label_names is not None:
                    postfix = self.label_names[index]
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

    def on_batch_end(self,outputs,seq_data,masks,**kwargs):
        labels = self.inference(seq_data.answers,masks)
        #N,C,L
        data = contagion_matrix(outputs,labels,masks)
        for answer_index in range(self.label_num):
            for predict_index in range(self.label_num):
                self._data[answer_index][predict_index] += data[answer_index][predict_index]

    @property
    def data(self):
        return {self.prefix+'contagion_matrix':self._data}

    def _reset(self):
        self._data = [[0]*self.label_num for _ in range(self.label_num)]
        
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

    def on_batch_end(self,outputs,seq_data,**kwargs):
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