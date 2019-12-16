import time
import abc
import json
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from .worker import TrainWorker,TestWorker
from .tensorboard_writer import TensorboardWriter
from .callback import CategoricalMetric,TensorboardCallback
from .callback import GFFCompare,SeqFigCallback, Callbacks,ContagionMatrix
from .inference import basic_inference
from ..process.data_processor import AnnSeqProcessor
from ..utils.utils import split,get_subdict,create_folder
from ..genome_handler.utils import ann_count
from ..utils.seq_converter import SeqConverter
from ..genome_handler import ann_seq_processor
from .data_generator import SeqGenerator
from .checkpoint import use_checkpoint

class SeqAnnEngine(metaclass=abc.ABCMeta):
    def __init__(self,channel_order):
        self._settings = {}
        self._path = None
        self._writer = None
        self._train_writer = self._val_writer = self._test_writer = None
        self.is_verbose_visible = True
        self._channel_order = channel_order
        self._ann_types = self._channel_order

    def update_settings(self,key,params):
        params = dict(params)
        if 'self' in params.keys():
            del params['self']
        self._settings[key] = params

    @property
    def ann_types(self):
        if self._ann_types is not None:
            return self._ann_types
        else:
            raise Exception("Object has not set train_ann_seqs or test_ann_seqs yet.")

    def get_seq_fig(self,seq,ann_seq,color_settings=None,prefix=None):
        if self._writer is None:
            raise Exception("Writer must be set first")
        seq = SeqConverter().seq2vecs(seq)
        seq = np.transpose(np.array([seq]),[0,2,1])
        seq = torch.from_numpy(seq).type('torch.FloatTensor').cuda()
        ann_seq = [ann_seq_processor.seq2vecs(ann_seq,self.ann_types)]
        color_settings = color_settings or {'other':'blue','exon':'red','intron':'yellow'}
        colors=[color_settings[type_]for type_ in self.ann_types]
        seq_fig = SeqFigCallback(self._writer,seq,ann_seq,prefix=prefix,label_names=self.ann_types,colors=colors)
        return seq_fig

    def set_root(self,path,with_train=True,with_val=True,with_test=True,
                 create_tensorboard=True):
        self.update_settings('set_root',locals())
        self._path = path
        create_folder(self._path)
        if create_tensorboard:    
            self._writer = TensorboardWriter(SummaryWriter(path))
        if with_train:
            fp = os.path.join(path,"train")
            create_folder(fp)
            if create_tensorboard:
                self._train_writer = TensorboardWriter(SummaryWriter(fp))
        if with_val:
            fp = os.path.join(path,"val")
            create_folder(fp)
            if create_tensorboard:
                self._val_writer = TensorboardWriter(SummaryWriter(fp))
        if with_test:
            fp = os.path.join(path,"test")
            create_folder(fp)
            if create_tensorboard:
                self._test_writer = TensorboardWriter(SummaryWriter(fp))

    def _record_ann_count(self,name,ann_seqs):
        if ann_seqs is not None:
            if ann_seqs.is_empty():
                raise Exception("The {} has no data inside".format(name))
            count = ann_count(ann_seqs)
            print(len(ann_seqs),count)
            count_ = np.array([int(val) for val in count.values()])
            if (count_==0).any():
                raise Exception("There are some annotations missing is the dataset")
            self.update_settings(name,count)
                
    def _add_gff_compare(self,ann_vec2info_converter,callbacks,prefix=None):
        path = self._path
        if path is not None and prefix is not None:
            path = os.path.join(path,prefix)
        gff_compare = GFFCompare(ann_vec2info_converter,path,prefix=prefix)
        callbacks.add(gff_compare)

    def _create_categorical_metric(self,prefix=None):
        metric = CategoricalMetric(len(self.ann_types),
                                   label_names=self.ann_types,
                                   prefix=prefix)
        return metric

    def _create_contagion_matrix(self,prefix=None):
        metric = ContagionMatrix(len(self.ann_types),
                                 label_names=self.ann_types,
                                 prefix=prefix)
        return metric

    def _create_default_train_callbacks(self,with_val=True):
        train_callbacks = Callbacks()
        train_metric = self._create_categorical_metric()
        train_matrix = self._create_contagion_matrix()
        train_callbacks.add([train_metric,train_matrix])
        val_callbacks=None
        if with_val:
            val_callbacks = Callbacks()
            val_metric = self._create_categorical_metric(prefix='val')
            val_matrix = self._create_contagion_matrix(prefix='val')
            val_callbacks.add([val_metric,val_matrix])
        return train_callbacks,val_callbacks

    def _create_default_test_callbacks(self):
        callbacks = Callbacks()
        test_metric = self._create_categorical_metric(prefix='test')
        test_matrix = self._create_contagion_matrix(prefix='test')
        callbacks.add([test_metric,test_matrix])
        return callbacks

    def _update_common_setting(self):
        self.update_settings('setting',{'ann_types':self._ann_types,
                                        'channel_order':self._channel_order})
    
    def _update_ann_seqs_count(self,name,origin_count,filtered_count):
        self.update_settings(name,{'origin count':origin_count,
                                   'filtered count':filtered_count})
    
    def _add_tensorboard_callback(self,writer,callbacks,add_grad=False):
        if writer is not None and callbacks is not None:
            tensorboard = TensorboardCallback(writer)
            tensorboard.do_add_grad = add_grad
            callbacks.add(tensorboard)
            
    def _process_data(self,raw_data):
        keys = list(raw_data.keys())
        data = AnnSeqProcessor(self._channel_order).process(raw_data)
        for key in keys:
            self._update_ann_seqs_count(key,len(raw_data[key]['answers'].ids),len(data[key]['ids']))
            self._record_ann_count('{}_ann_counut'.format(key),raw_data[key]['answers'])
        return data
    
    def train(self,model,executor,train_data,val_data=None,
              epoch=None,batch_size=None,other_callbacks=None,
              augmentation_max=None,add_grad=True,checkpoint_kwargs=None):
        self._update_common_setting()
        checkpoint_kwargs = checkpoint_kwargs or {}
        other_callbacks = other_callbacks or Callbacks()
        epoch = epoch or 100
        self.update_settings('train_setting',{'epoch':epoch,'batch_size':batch_size,
                                              'augmentation_max':augmentation_max,
                                              'add_grad':add_grad,
                                              'checkpoint_kwargs':checkpoint_kwargs})
        #Set data
        train_seqs,train_ann_seqs = train_data
        with_val = val_data is not None
        if with_val:
            val_seqs,val_ann_seqs = val_data
        else:
            val_seqs = val_ann_seqs = None
        #Set callbacks and writer
        train_callbacks, val_callbacks = self._create_default_train_callbacks(with_val)
        epoch_start,checkpoint = use_checkpoint(self._path,**checkpoint_kwargs)
        other_callbacks.add(checkpoint)
        self._add_tensorboard_callback(self._train_writer,train_callbacks,add_grad=add_grad)
        self._add_tensorboard_callback(self._val_writer,val_callbacks,add_grad=add_grad)

        #Create worker    
        train_gen = SeqGenerator(batch_size=batch_size,augmentation_max=augmentation_max)
        val_gen = SeqGenerator(batch_size=batch_size,shuffle=False)
            
        #Process data
        raw_data = {'training':{'inputs':train_seqs,'answers':train_ann_seqs}}
        if val_ann_seqs is not None:
             raw_data['validation'] = {'inputs':val_seqs,'answers':val_ann_seqs}
        data = self._process_data(raw_data)
            
        #Create worker
        worker = TrainWorker(model,data,
                             train_generator=train_gen,val_generator=val_gen,
                             executor=executor,train_callbacks=train_callbacks,
                             val_callbacks=val_callbacks,other_callbacks=other_callbacks,
                             writer=self._writer,epoch=epoch,path=self._path,
                             epoch_start=epoch_start)
        worker.is_verbose_visible = self.is_verbose_visible
        #Save setting
        if self._path is not None:
            setting_path = os.path.join(self._path,"train_facade_setting.json")
            model_config_path = os.path.join(self._path,"model_config.json")
            model_component_path = os.path.join(self._path,"model_component.txt")
            exec_config_path = os.path.join(self._path,"executor_config.json")
            with open(setting_path,'w') as fp:
                json.dump(self._settings,fp, indent=4)
            with open(model_config_path,'w') as fp:
                json.dump(model.get_config(),fp, indent=4)
            with open(model_component_path,'w') as fp:
                fp.write(str(model))
            with open(exec_config_path,'w') as fp:
                json.dump(executor.get_config(),fp, indent=4)

        #Execute worker
        worker.work()
        return worker

    def test(self,model,executor,data,batch_size=None,ann_vec2info_converter=None,callbacks=None):
        self._update_common_setting()
        self.update_settings('test_setting',{'batch_size':batch_size})
        callbacks = callbacks or Callbacks()
        test_callbacks = self._create_default_test_callbacks()
        callbacks.add(test_callbacks)
        self._add_tensorboard_callback(self._test_writer,callbacks)
        if ann_vec2info_converter is not None:
            self._add_gff_compare(ann_vec2info_converter,callbacks,prefix='test')
        generator = SeqGenerator(batch_size=batch_size,shuffle=False)
        test_seqs,test_ann_seqs = data
        raw_data = {'testing':{'inputs':test_seqs,'answers':test_ann_seqs}}
        data = self._process_data(raw_data)
        
        worker = TestWorker(model,data,
                            generator=generator,
                            callbacks=callbacks,
                            executor=executor,
                            path=self._path)

        if self._path is not None:
            fp = os.path.join(self._path,"test_facade_setting.json")
            with open(fp,'w') as fp:
                json.dump(self._settings,fp, indent=4)
        worker.work()
        return worker.result
