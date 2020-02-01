import abc
import numpy as np
import torch
import os
from ..utils.utils import create_folder, write_json
from ..utils.seq_converter import SeqConverter
from ..genome_handler.ann_genome_processor import class_count
from ..genome_handler.ann_seq_processor import seq2vecs
from .data_processor import AnnSeqProcessor
from .utils import param_num
from .worker import TrainWorker,TestWorker
from .tensorboard_writer import TensorboardWriter
from .callback import CategoricalMetric,TensorboardCallback
from .callback import SeqFigCallback, Callbacks,ContagionMatrix
from .signal_handler import build_signal_handler
from .data_generator import SeqGenerator

class SeqAnnEngine(metaclass=abc.ABCMeta):
    def __init__(self,channel_order,shuffle_train_data=True,is_verbose_visible=True):
        self._settings = {}
        self._path = None
        self._writer = None
        self._train_writer = self._val_writer = self._test_writer = None
        self.is_verbose_visible = is_verbose_visible
        self._channel_order = channel_order
        self._ann_types = self._channel_order
        self._shuffle_train_data = shuffle_train_data

    def update_settings(self,key,params):
        params = dict(params)
        if 'self' in params.keys():
            del params['self']
        self._settings[key] = params

    def print_verbose(self,info,*args):
        if self.is_verbose_visible:
            print(info,*args)
        
    @property
    def ann_types(self):
        if self._ann_types is not None:
            return self._ann_types
        else:
            raise Exception("Object has not set train_ann_seqs or test_ann_seqs yet.")

    def get_seq_fig(self,seq,ann_seq,color_settings,prefix=None):
        if self._writer is None:
            raise Exception("Writer must be set first")
        seq = SeqConverter().seq2vecs(seq)
        seq = np.transpose(np.array([seq]),[0,2,1])
        seq = torch.from_numpy(seq).type('torch.FloatTensor').cuda()
        ann_seq = [seq2vecs(ann_seq,self.ann_types)]
        colors=[color_settings[type_] for type_ in self.ann_types]
        seq_fig = SeqFigCallback(self._writer,seq,ann_seq,prefix=prefix,label_names=self.ann_types,colors=colors)
        return seq_fig

    def set_root(self,path,with_train=True,with_val=True,with_test=True,
                 create_tensorboard=True):
        self.update_settings('set_root',locals())
        self._path = path
        create_folder(self._path)
        if create_tensorboard:    
            self._writer = TensorboardWriter(path)
        if with_train:
            path_ = os.path.join(path,"train")
            create_folder(path_)
            if create_tensorboard:
                self._train_writer = TensorboardWriter(path_)
        if with_val:
            path_ = os.path.join(path,"val")
            create_folder(path_)
            if create_tensorboard:
                self._val_writer = TensorboardWriter(path_)
        if with_test:
            path_ = os.path.join(path,"test")
            create_folder(path_)
            if create_tensorboard:
                self._test_writer = TensorboardWriter(path_)

    def _record_ann_count(self,name,ann_seqs):
        if ann_seqs is not None:
            if ann_seqs.is_empty():
                raise Exception("The {} has no data inside".format(name))
            count = class_count(ann_seqs)
            self.print_verbose(len(ann_seqs),count)
            for key,value in count.items():
                if int(value)==0:
                    raise Exception("The {} is missing in the dataset".format(key))
            self.update_settings(name,count)
                
    def _add_signal_handler(self,ann_vec_gff_converter,region_table_path,
                            answer_gff_path,callbacks,prefix=None):
        path = self._path
        if path is not None and prefix is not None:
            path = os.path.join(path,prefix)
           
        verified_paths = [region_table_path,answer_gff_path]
        if all([verified_path is not None for verified_path in verified_paths]):
            signal_handler = build_signal_handler(path,region_table_path,
                                                  answer_gff_path,
                                                  ann_vec_gff_converter,
                                                  prefix=prefix)

            callbacks.add(signal_handler)

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
    
    def _create_train_data_gen(self,batch_size,augmentation_max):
        train_gen = SeqGenerator(batch_size=batch_size,
                                 augmentation_max=augmentation_max,
                                 shuffle=self._shuffle_train_data)
        return train_gen
    
    def _create_test_data_gen(self,batch_size):
        test_gen = SeqGenerator(batch_size=batch_size,shuffle=False)
        return test_gen
    
    def train(self,model,executor,train_data,val_data=None,
              epoch=None,batch_size=None,other_callbacks=None,
              augmentation_max=None,add_grad=True,checkpoint_kwargs=None):

        if self._path is not None:
            with open(os.path.join(self._path,'param_num.txt'),"w") as fp:
                fp.write("Required-gradient parameters number:{}".format(param_num(model)))
        
        self._update_common_setting()
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
        self._add_tensorboard_callback(self._train_writer,train_callbacks,add_grad=add_grad)
        self._add_tensorboard_callback(self._val_writer,val_callbacks,add_grad=add_grad)

        #Create worker    
        train_gen = self._create_train_data_gen(batch_size,augmentation_max)
        val_gen = self._create_test_data_gen(batch_size)
            
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
                             checkpoint_kwargs=checkpoint_kwargs)
        worker.is_verbose_visible = self.is_verbose_visible
        #Save setting
        if self._path is not None:
            setting_path = os.path.join(self._path,"train_facade_setting.json")
            model_config_path = os.path.join(self._path,"model_config.json")
            model_component_path = os.path.join(self._path,"model_component.txt")
            exec_config_path = os.path.join(self._path,"executor_config.json")
            
            write_json(self._settings,setting_path)
            write_json(model.get_config(),model_config_path)
            write_json(str(model),model_component_path)
            write_json(executor.get_config(),exec_config_path)

        #Execute worker
        worker.work()
        return worker

    def test(self,model,executor,data,batch_size=None,
             ann_vec_gff_converter=None,region_table_path=None,
             answer_gff_path=None,callbacks=None):

        self._update_common_setting()
        self.update_settings('test_setting',{'batch_size':batch_size})
        callbacks = callbacks or Callbacks()
        test_callbacks = self._create_default_test_callbacks()
        callbacks.add(test_callbacks)
        self._add_tensorboard_callback(self._test_writer,callbacks)
        if ann_vec_gff_converter is not None and region_table_path is not None:
            self._add_signal_handler(ann_vec_gff_converter,region_table_path,
                                     answer_gff_path,callbacks,prefix='test')
        generator = self._create_test_data_gen(batch_size)
        test_seqs,test_ann_seqs = data
        raw_data = {'testing':{'inputs':test_seqs,'answers':test_ann_seqs}}
        data = self._process_data(raw_data)
        
        worker = TestWorker(model,data,
                            generator=generator,
                            callbacks=callbacks,
                            executor=executor,
                            path=self._path)

        if self._path is not None:
            path = os.path.join(self._path,"test_facade_setting.json")
            write_json(self._settings,path)

        worker.work()
        return worker.result
