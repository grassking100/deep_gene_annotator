import torch
import random
import numpy as np
from tensorboardX import SummaryWriter
from .worker import TrainWorker,TestWorker
from .callback import CategoricalMetric,TensorboardCallback,TensorboardWriter
from .callback import EarlyStop,GFFCompare,SeqFigCallback, Callbacks
from .executer import ModelExecutor
from ..process.pipeline import Pipeline 
from ..process.data_processor import AnnSeqProcessor
from ..utils.utils import split,get_subdict
from ..genome_handler.utils import get_subseqs,ann_count
from ..data_handler.seq_converter import SeqConverter
from ..genome_handler import ann_seq_processor
from ..process.data_generator import SeqGenerator

class SeqAnnFacade:
    def __init__(self):
        self._settings = {}
        self.train_seqs = None
        self.val_seqs = None
        self.test_seqs = None
        self.train_ann_seqs = None
        self.val_ann_seqs = None
        self.test_ann_seqs = None
        self._other_callbacks = Callbacks()
        self._model_executor = ModelExecutor()
        self._path = None
        self._writer = None
        self._train_writer = None
        self._val_writer = None
        self._test_writer = None
        self.simplify_map = None

    def update_settings(self,key,params):
        if key not in self._settings.keys():
            params = dict(params)
            if 'self' in params.keys():
                del params['self']
            self._settings[key] = params

    def clean_callbacks(self):
        self._other_callbacks.clean()

    def set_optimizer(self,**kwargs):
        self.update_settings('set_optimizer',kwargs)
        self._model_executor = ModelExecutor()
        for key,value in kwargs.items():
            if hasattr(self._model_executor,key):
                if value is not None:
                    setattr(self._model_executor,key,value)
            else:
                raise Exception(str(self._model_executor)+" has no attribute "+key)

    @property
    def ann_types(self):
        if self.train_ann_seqs is not None:
            return self.train_ann_seqs.ANN_TYPES
        elif self.test_ann_seqs is not None:
            return self.test_ann_seqs.ANN_TYPES
        else:
            raise Exception("Object has not set train_ann_seqs or test_ann_seqs yet.")

    def assign_data_by_random(self,seqs,ann_seqs,ratio=None):
        self.update_settings('assign_data_by_random',{'ratio':ratio})
        ratio = ratio or [0.7,0.2,0.1]
        ids = list(seqs.keys())
        random.shuffle(ids)
        sub_ids = split(ids,ratio)
        self.train_seqs = get_subdict(sub_ids[0],seqs)
        self.train_ann_seqs = get_subseqs(sub_ids[0],ann_seqs)
        self.val_seqs = get_subdict(sub_ids[1],seqs)
        self.val_ann_seqs = get_subseqs(sub_ids[1],ann_seqs)
        self.test_seqs = get_subdict(sub_ids[2],seqs)
        self.test_ann_seqs = get_subseqs(sub_ids[2],ann_seqs)
        ann_seq_list = [self.train_ann_seqs,self.val_ann_seqs,self.test_ann_seqs]
        self._validate_ann_seqs(ann_seq_list)

    def _validate_ann_seqs(self,ann_seq_list):
        for ann_seqs in ann_seq_list:
            if not ann_seqs.is_empty():
                count = ann_count(ann_seqs)
                print(len(ann_seqs),count)
                count = np.array([int(val) for val in count.values()])
                if (count==0).any():
                    raise Exception("There are some annotations missing is the dataset")

    def add_early_stop(self,target=None,optimize_min=None,patient=None,
                       saved_best_weights=None,restored_best_weights=None):
        self.update_settings('add_early_stop',locals())
        early_stop = EarlyStop()
        for key,value in param.item():
            if hasattr(early_stop,key):
                if value is not None:
                    setattr(early_stop,key,vlue)
            else:
                raise Exception(str(early_stop)+" has no attribute "+key)
        self._other_callbacks.append(early_stop)

    def add_seq_fig(self,seq,ann_seq,colors_settings=None,prefix=None,inference=None):
        if self._writer is None:
            raise Exception("Writer must be set first")
        params = locals()
        del params['seq']
        del params['ann_seq']
        self.update_settings('add_seq_fig',params)
        seq = SeqConverter().seq2vecs(seq)
        seq = np.transpose(np.array([seq]),[0,2,1])
        seq = torch.from_numpy(seq).type('torch.FloatTensor').cuda()
        ann_seq = ann_seq_processor.seq2vecs(ann_seq)
        colors_settings = colors_settings or {'other':'blue','exon':'red','intron':'yellow'}
        colors=[colors_settings[type_]for type_ in self.ann_types]
        seq_fig = SeqFigCallback(self._writer,seq,ann_seq,inference=inference)
        seq_fig.class_names=self.ann_types
        seq_fig.colors=colors
        seq_fig.prefix=prefix
        self._other_callbacks.add_callbacks(seq_fig)

    def set_path(self,path,with_train=True,with_val=True,with_test=False):
        self.update_settings('set_path',locals())
        self._path = path
        self._writer = TensorboardWriter(SummaryWriter(path))
        if with_train:
            self._train_writer = TensorboardWriter(SummaryWriter(path+"/train"))
        if with_val:
            self._val_writer = TensorboardWriter(SummaryWriter(path+"/val"))
        if with_test:
            self._test_writer = TensorboardWriter(SummaryWriter(path+"/test"))

    def _create_gff_compare(self,path):
        if self.simplify_map is None:
            raise Exception("The simplify_map must be set first")
        gff_compare = GFFCompare(self.ann_types,path,simplify_map=self.simplify_map)
        return gff_compare

    def _create_categorical_metric(self,prefix=None):
        metric = CategoricalMetric()
        metric.class_num=len(self.ann_types)
        metric.class_names=self.ann_types
        metric.prefix=prefix
        return metric

    def _create_default_train_callbacks(self,path=None):
        train_callbacks = Callbacks()
        val_callbacks = Callbacks()
        train_metric = self._create_categorical_metric()
        val_metric = self._create_categorical_metric(prefix='val')  
        train_callbacks.add_callbacks(train_metric)
        val_callbacks.add_callbacks(val_metric)
        if path is not None:
            val_callbacks.add_callbacks(self._create_gff_compare(path))
        return train_callbacks,val_callbacks

    def _create_default_test_callbacks(self,path=None):
        callbacks = Callbacks()
        test_metric = self._create_categorical_metric(prefix='test')  
        callbacks.add_callbacks(test_metric)
        if path is not None:
            callbacks.add_callbacks(self._create_gff_compare(path))
        return callbacks

    def train(self,model,epoch_num=100,batch_size=32):
        self.update_settings('train',locals())
        self.update_settings('train_model_space',model.settings)
        train_callbacks, val_callbacks = self._create_default_train_callbacks(self._path)
        self._validate_ann_seqs([self.train_ann_seqs,self.val_ann_seqs])
        writers = [self._train_writer,self._val_writer,self._test_writer]
        callbacks_list = [train_callbacks,val_callbacks,self._other_callbacks]
        for writer,callback in zip(writers,callbacks_list):
            if writer is not None:
                tensorboard = TensorboardCallback(writer)
                callback.add_callbacks(tensorboard)
        if self._path is not None:
            with open(self._path+"/facade_train_setting.json",'w') as fp:
                fp.write(str(self._settings))
        train_gen = SeqGenerator()
        val_gen = SeqGenerator()
        train_gen.batch_size = val_gen.batch_size = batch_size
        worker = TrainWorker(train_generator=train_gen,val_generator=val_gen,
                             executor=self._model_executor,train_callbacks=train_callbacks,
                             val_callbacks=val_callbacks,other_callbacks=self._other_callbacks,
                             writer=self._writer,epoch_num=epoch_num)
        data_ = {'training':{'inputs':self.train_seqs,'answers':self.train_ann_seqs},
                 'validation':{'inputs':self.val_seqs,'answers':self.val_ann_seqs}}
        data = AnnSeqProcessor(data_,discard_invalid_seq=True).process()
        pipeline = Pipeline(model,data,worker)
        pipeline.path=self._path
        pipeline.execute()
        record = pipeline.result
        self.clean_callbacks()
        if self._path is not None:
            with open(self._path+"/train_record.json",'w') as fp:
                fp.write(str(record))
        return record

    def test(self,model,batch_size=32):
        self.update_settings('test',locals())
        self.update_settings('test_model_space',model.settings)
        self._validate_ann_seqs([self.test_ann_seqs])
        callbacks = self._create_default_test_callbacks(self._path)
        if self._test_writer is not None:
            tensorboard = self.TensorboardCallback(self._test_writer)
            callbacks.add_callbacks(gff_compare)
        if self._path is not None:
            with open(self._path+"/facade_test_setting.json",'w') as fp:
                fp.write(str(self._settings))
        gen = SeqGenerator()
        gen.batch_size = batch_size
        worker = TestWorker(generator = gen,callbacks=callbacks,executor=self._model_executor,writer=writer)
        data_ = {'testing':{'inputs':self.test_seqs,'answers':self.test_ann_seqs}}
        data = AnnSeqProcessor(data_,discard_invalid_seq=True).process()
        pipeline = Pipeline(model,data,worker)
        pipeline.path=self._path
        pipeline.execute()
        record = pipeline.result
        if self._path is not None:
            with open(self._path+"/test_record.json",'w') as fp:
                fp.write(str(record))
        return record
