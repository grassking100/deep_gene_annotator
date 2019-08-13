import json
import random
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from .worker import TrainWorker,TestWorker
from .tensorboard_writer import TensorboardWriter
from .callback import CategoricalMetric,TensorboardCallback
from .callback import GFFCompare,SeqFigCallback, Callbacks,ContagionMatrix
from .executer import BasicExecutor
from ..process.pipeline import Pipeline
from ..process.data_processor import AnnSeqProcessor
from ..utils.utils import split,get_subdict
from ..genome_handler.utils import get_subseqs,ann_count
from ..utils.seq_converter import SeqConverter
from ..genome_handler import ann_seq_processor
from ..process.data_generator import SeqGenerator

def split_data_by_random(seqs,ann_seqs,ratio=None):
    ratio = ratio or [0.7,0.2,0.1]
    ids = list(seqs.keys())
    random.shuffle(ids)
    sub_ids = split(ids,ratio)
    train_seqs = get_subdict(sub_ids[0],seqs)
    train_ann_seqs = get_subseqs(sub_ids[0],ann_seqs)
    val_seqs = get_subdict(sub_ids[1],seqs)
    val_ann_seqs = get_subseqs(sub_ids[1],ann_seqs)
    test_seqs = get_subdict(sub_ids[2],seqs)
    test_ann_seqs = get_subseqs(sub_ids[2],ann_seqs)
    seqs = train_seqs, val_seqs, test_seqs
    ann_seqs = train_ann_seqs, val_ann_seqs, test_ann_seqs
    return seqs,ann_seqs

class SeqAnnFacade:
    def __init__(self):
        self._settings = {}
        self.train_seqs = self.val_seqs = self.test_seqs = None
        self.train_ann_seqs = self.val_ann_seqs = self.test_ann_seqs = None
        self.other_callbacks = Callbacks()
        self.executor = BasicExecutor()
        self._path = None
        self._writer = None
        self._train_writer = self._val_writer = self._test_writer = None
        self.simplify_map = None
        self.alt = False
        self.use_gffcompare = True
        self.alt_num = 0
        self.is_verbose_visible = True

    def update_settings(self,key,params):
        params = dict(params)
        if 'self' in params.keys():
            del params['self']
        self._settings[key] = params

    @property
    def ann_types(self):
        if self.train_ann_seqs is not None:
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
        ann_seq = [ann_seq_processor.seq2vecs(ann_seq)]
        color_settings = color_settings or {'other':'blue','exon':'red','intron':'yellow'}
        colors=[color_settings[type_]for type_ in self.ann_types]
        seq_fig = SeqFigCallback(self._writer,seq,ann_seq,prefix=prefix,label_names=self.ann_types,colors=colors)
        self.other_callbacks.add(seq_fig)

    def set_root(self,path,with_train=True,with_val=True,with_test=True):
        self.update_settings('set_root',locals())
        self._path = path
        self._writer = TensorboardWriter(SummaryWriter(path))
        if with_train:
            fp = os.path.join(path,"train")
            self._train_writer = TensorboardWriter(SummaryWriter(fp))
        if with_val:
            fp = os.path.join(path,"val")
            self._val_writer = TensorboardWriter(SummaryWriter(fp))
        if with_test:
            fp = os.path.join(path,"test")
            self._test_writer = TensorboardWriter(SummaryWriter(fp))

    def _create_gff_compare(self,path,prefix=None):
        if self.simplify_map is None:
            raise Exception("The simplify_map must be set first")
        gff_compare = GFFCompare(self.ann_types,path,simplify_map=self.simplify_map,prefix=prefix)
        gff_compare.extractor.alt = self.alt
        gff_compare.extractor.alt_num = self.alt_num
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
        train_callbacks.add(train_metric)
        train_callbacks.add(train_matrix)
        val_callbacks.add(val_metric)
        val_callbacks.add(val_matrix)
        if path is not None:
            if self.use_gffcompare:
                gff_compare = self._create_gff_compare(path,prefix='val')
                val_callbacks.add(gff_compare)
        return train_callbacks,val_callbacks

    def _create_default_test_callbacks(self,path=None):
        callbacks = Callbacks()
        test_metric = self._create_categorical_metric(prefix='test')
        test_matrix = self._create_contagion_matrix(prefix='test')
        callbacks.add(test_metric)
        callbacks.add(test_matrix)
        if path is not None:
            if self.use_gffcompare:
                gff_compare = self._create_gff_compare(path,prefix='test')
                callbacks.add(gff_compare)
        return callbacks

    def train(self,model,epoch=100,batch_size=32,augmentation_max=None):
        self.update_settings('train_setting',{'epoch':epoch,'batch_size':batch_size})
        fp = os.path.join(self._path,"val")
        train_callbacks, val_callbacks = self._create_default_train_callbacks(fp)
        train_ann_count = self._validate_ann_seqs([self.train_ann_seqs])
        self.update_settings('train_ann_counut',train_ann_count[0])
        
        if self.val_ann_seqs is not None:
            val_ann_count = self._validate_ann_seqs([self.val_ann_seqs])
            self.update_settings('val_ann_count',val_ann_count[0])
        writers = [self._train_writer,self._val_writer,None]
        callbacks_list = [train_callbacks,val_callbacks,self.other_callbacks]
        for writer,callback in zip(writers,callbacks_list):
            if writer is not None and callback is not None:
                tensorboard = TensorboardCallback(writer)
                callback.add(tensorboard)        
        train_gen = SeqGenerator(batch_size,augmentation_max=augmentation_max)
        val_gen = SeqGenerator(batch_size)
        if self.val_ann_seqs is None:
            val_callbacks = None
        worker = TrainWorker(train_generator=train_gen,val_generator=val_gen,
                             executor=self.executor,train_callbacks=train_callbacks,
                             val_callbacks=val_callbacks,other_callbacks=self.other_callbacks,
                             writer=self._writer,epoch=epoch)
        worker.is_verbose_visible = self.is_verbose_visible
        data_ = {'training':{'inputs':self.train_seqs,'answers':self.train_ann_seqs}}
        if self.val_ann_seqs is not None:
             data_['validation'] = {'inputs':self.val_seqs,'answers':self.val_ann_seqs}
        data = AnnSeqProcessor(data_,discard_invalid_seq=True).process()
        origin_train_num = len(self.train_ann_seqs.ids)
        filtered_train_num = len(data['training']['extra']['ids'])
        self.update_settings('train_seq',{'origin count':origin_train_num,
                                          'filtered count':filtered_train_num})
        if self.val_ann_seqs is not None:
            origin_val_num = len(self.val_ann_seqs.ids)
            filtered_val_num  = len(data['validation']['extra']['ids'])
            self.update_settings('val_seq',{'origin count':origin_val_num,
                                            'filtered count':filtered_val_num})

        pipeline = Pipeline(model,data,worker,path=self._path)
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
        pipeline.execute()
        if self._path is not None:
            record_path = os.path.join(self._path,"train_record.json")
            with open(record_path,'w') as fp:
                json.dump(worker.result,fp, indent=4)
            if worker.best_epoch is None:
                model_path =  os.path.join(self.path,'last_model.pth')
                torch.save(worker.model.state_dict(),model_path)
            else:    
                best_path = os.path.join(self._path,"best_record.json")
                best_result = {'best_epoch':worker.best_epoch,
                               'best_result':worker.best_result}
            with open(best_path,'w') as fp:
                json.dump(best_result,fp)
                 
        return worker.result

    def test(self,model,batch_size=32):
        self.update_settings('test_setting',{'batch_size':batch_size})
        test_ann_count = self._validate_ann_seqs([self.test_ann_seqs])[0]
        self.update_settings('test_ann_count',test_ann_count)
        fp = os.path.join(self._path,"test")
        callbacks = self._create_default_test_callbacks(fp)
        if self._test_writer is not None:
            tensorboard = TensorboardCallback(self._test_writer)
            callbacks.add(tensorboard)
        gen = SeqGenerator(batch_size)
        worker = TestWorker(generator = gen,callbacks=callbacks,executor=self.executor)
        data_ = {'testing':{'inputs':self.test_seqs,'answers':self.test_ann_seqs}}
        data = AnnSeqProcessor(data_,discard_invalid_seq=True).process()
        origin_num  = len(self.test_ann_seqs.ids)
        filtered_num = len(data['testing']['extra']['ids'])
        self.update_settings('test_seq',{'origin count':origin_num,
                                         'filtered count':filtered_num})
        pipeline = Pipeline(model,data,worker,path=self._path)
        if self._path is not None:
            fp = os.path.join(self._path,"facade_test_setting.json")
            with open(fp,'w') as fp:
                json.dump(self._settings,fp, indent=4)
        pipeline.execute()
        if self._path is not None:
            fp = os.path.join(self._path,"test_record.json")
            with open(fp,'w') as fp:
                json.dump(worker.result,fp, indent=4)
        return worker.result
