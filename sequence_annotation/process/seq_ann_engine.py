import time
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
from .executor import BasicExecutor
from ..process.data_processor import AnnSeqProcessor
from ..utils.utils import split,get_subdict,create_folder
from ..genome_handler.utils import ann_count
from ..utils.seq_converter import SeqConverter
from ..genome_handler import ann_seq_processor
from .data_generator import SeqGenerator

class SeqAnnEngine:
    def __init__(self,ann_types=None,channel_order=None):
        self._settings = {}
        self.train_seqs = self.val_seqs = self.test_seqs = None
        self.train_ann_seqs = self.val_ann_seqs = self.test_ann_seqs = None
        self.train_callbacks = Callbacks()
        self.val_callbacks = Callbacks()
        self.other_callbacks = Callbacks()
        self.executor = BasicExecutor()
        self._path = None
        self._writer = None
        self._train_writer = self._val_writer = self._test_writer = None
        self.use_gffcompare = True
        self.is_verbose_visible = True
        self.fix_boundary = False
        self._ann_types = ann_types
        self._channel_order = channel_order or self.ann_types
        self.ann_vec2info_converter = None

    def update_settings(self,key,params):
        params = dict(params)
        if 'self' in params.keys():
            del params['self']
        self._settings[key] = params

    @property
    def ann_types(self):
        if self._ann_types is not None:
            return self._ann_types
        elif self.train_ann_seqs is not None:
            return self.train_ann_seqs.ANN_TYPES
        elif self.test_ann_seqs is not None:
            return self.test_ann_seqs.ANN_TYPES
        else:
            raise Exception("Object has not set train_ann_seqs or test_ann_seqs yet.")

    def _validate_ann_seqs(self,ann_seq_list):
        counts = []
        for ann_seqs in ann_seq_list:
            if not ann_seqs.is_empty():
                count = ann_count(ann_seqs)
                counts.append(count)
                print(len(ann_seqs),count)
                count = np.array([int(val) for val in count.values()])
                if (count==0).any():
                    raise Exception("There are some annotations missing is the dataset")
        return counts

    def add_seq_fig(self,seq,ann_seq,color_settings=None,prefix=None):
        if self._writer is None:
            raise Exception("Writer must be set first")
        seq = SeqConverter().seq2vecs(seq)
        seq = np.transpose(np.array([seq]),[0,2,1])
        seq = torch.from_numpy(seq).type('torch.FloatTensor').cuda()
        ann_seq = [ann_seq_processor.seq2vecs(ann_seq,self.ann_types)]
        color_settings = color_settings or {'other':'blue','exon':'red','intron':'yellow'}
        colors=[color_settings[type_]for type_ in self.ann_types]
        seq_fig = SeqFigCallback(self._writer,seq,ann_seq,prefix=prefix,label_names=self.ann_types,colors=colors)
        self.other_callbacks.add(seq_fig)

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

    def _create_gff_compare(self,path,prefix=None):
        if self.ann_vec2info_converter is None:
            raise Exception("The ann_vec2info_converter must be set first")            
        gff_compare = GFFCompare(self.ann_vec2info_converter,path,
                                 fix_boundary=self.fix_boundary,prefix=prefix)
        return gff_compare

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


    def _create_default_train_callbacks(self,path=None):
        train_callbacks = Callbacks()
        val_callbacks = Callbacks()
        train_metric = self._create_categorical_metric()
        val_metric = self._create_categorical_metric(prefix='val')
        train_matrix = self._create_contagion_matrix()
        val_matrix = self._create_contagion_matrix(prefix='val')
        train_callbacks.add([train_metric,train_matrix])
        val_callbacks.add([val_metric,val_matrix])
        if path is not None:
            if self.use_gffcompare:
                gff_compare = self._create_gff_compare(path,prefix='val')
                val_callbacks.add(gff_compare)
        return train_callbacks,val_callbacks

    def _create_default_test_callbacks(self,path=None):
        callbacks = Callbacks()
        test_metric = self._create_categorical_metric(prefix='test')
        test_matrix = self._create_contagion_matrix(prefix='test')
        callbacks.add([test_metric,test_matrix])
        if path is not None:
            if self.use_gffcompare:
                gff_compare = self._create_gff_compare(path,prefix='test')
                callbacks.add(gff_compare)
        return callbacks

    def update_common_setting(self):
        self.update_settings('setting',{'use_gffcompare':self.use_gffcompare,
                                        'fix_boundary':self.fix_boundary,
                                        'ann_types':self._ann_types,
                                        'channel_order':self._channel_order})
    
    def train(self,model,epoch=None,batch_size=None,augmentation_max=None,add_grad=True,epoch_start=None):
        self.update_common_setting()
        epoch = epoch or 100
        batch_size = batch_size or 32
        self.update_settings('train_setting',{'epoch':epoch,'batch_size':batch_size})
        #Set callbacks and writer
        fp = None
        if self._path is not None:
            fp = os.path.join(self._path,"val")
        train_callbacks, val_callbacks = self._create_default_train_callbacks(fp)
        train_callbacks.add(self.train_callbacks)
        val_callbacks.add(self.val_callbacks)
        callbacks_list = [train_callbacks,val_callbacks,self.other_callbacks]
        writers = [self._train_writer,self._val_writer,None]
        for writer,callback in zip(writers,callbacks_list):
            if writer is not None and callback is not None:
                tensorboard = TensorboardCallback(writer)
                tensorboard.do_add_grad = add_grad
                callback.add(tensorboard)
        
        #Validate ann_seqs and set val_callbacks
        train_ann_count = self._validate_ann_seqs([self.train_ann_seqs])
        self.update_settings('train_ann_counut',train_ann_count[0])
        if self.val_ann_seqs is not None:
            val_ann_count = self._validate_ann_seqs([self.val_ann_seqs])
            self.update_settings('val_ann_count',val_ann_count[0])
        else:
            val_callbacks = None
        #Create worker    
        train_gen = SeqGenerator(batch_size=batch_size,augmentation_max=augmentation_max)
        val_gen = SeqGenerator(batch_size=batch_size,shuffle=False)
            
        #Process data
        data_ = {'training':{'inputs':self.train_seqs,'answers':self.train_ann_seqs}}
        if self.val_ann_seqs is not None:
             data_['validation'] = {'inputs':self.val_seqs,'answers':self.val_ann_seqs}
        data = AnnSeqProcessor(data_,ann_types=self._channel_order).process()
        origin_train_num = len(self.train_ann_seqs.ids)
        filtered_train_num = len(data['training']['ids'])
        self.update_settings('train_seq',{'origin count':origin_train_num,
                                          'filtered count':filtered_train_num})
        if self.val_ann_seqs is not None:
            origin_val_num = len(self.val_ann_seqs.ids)
            filtered_val_num  = len(data['validation']['ids'])
            self.update_settings('val_seq',{'origin count':origin_val_num,
                                            'filtered count':filtered_val_num})
        #Create worker
        worker = TrainWorker(model,data,
                             train_generator=train_gen,val_generator=val_gen,
                             executor=self.executor,train_callbacks=train_callbacks,
                             val_callbacks=val_callbacks,other_callbacks=self.other_callbacks,
                             writer=self._writer,epoch=epoch,path=self._path,
                             epoch_start=epoch_start)
        worker.is_verbose_visible = self.is_verbose_visible
        #Save setting
        if self._path is not None:
            setting_path = os.path.join(self._path,"train_facade_setting.json")
            with open(setting_path,'w') as fp:
                json.dump(self._settings,fp, indent=4)
            config_path = os.path.join(self._path,"model_config.json")
            with open(config_path,'w') as fp:
                json.dump(model.get_config(),fp, indent=4)
            model_component_path = os.path.join(self._path,"model_component.txt")
            with open(model_component_path,'w') as fp:
                fp.write(str(model))
            config_path = os.path.join(self._path,"executor_config.json")
            with open(config_path,'w') as fp:
                json.dump(self.executor.get_config(),fp, indent=4)

        #Execute worker
        worker.work()
        return worker

    def test(self,model,batch_size=None,use_default_callback=True):
        self.update_common_setting()
        batch_size = batch_size or 32
        self.update_settings('test_setting',{'batch_size':batch_size})
        test_ann_count = self._validate_ann_seqs([self.test_ann_seqs])[0]
        self.update_settings('test_ann_count',test_ann_count)
        fp = None
        if self._path is not None:
            fp = os.path.join(self._path,"test")
        if use_default_callback:
            callbacks = self._create_default_test_callbacks(fp)
        else:
            callbacks = Callbacks()
        callbacks.add(self.other_callbacks)
        if self._test_writer is not None:
            tensorboard = TensorboardCallback(self._test_writer)
            callbacks.add(tensorboard)
        generator = SeqGenerator(batch_size=batch_size,shuffle=False)

        data_ = {'testing':{'inputs':self.test_seqs,'answers':self.test_ann_seqs}}
        data = AnnSeqProcessor(data_,ann_types=self._channel_order).process()
        origin_num  = len(self.test_ann_seqs.ids)
        filtered_num = len(data['testing']['ids'])
        self.update_settings('test_seq',{'origin count':origin_num,
                                         'filtered count':filtered_num})
        worker = TestWorker(model,data,
                            generator=generator,
                            callbacks=callbacks,
                            executor=self.executor,
                            path=self._path)
        if self._path is not None:
            fp = os.path.join(self._path,"test_facade_setting.json")
            with open(fp,'w') as fp:
                json.dump(self._settings,fp, indent=4)
        worker.work()
        return worker.result
