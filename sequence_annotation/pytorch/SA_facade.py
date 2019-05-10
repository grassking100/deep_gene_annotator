from torch import nn
import torch
import random
import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
from .worker import TrainWorker,TestWorker
from .CRF import BatchCRFLoss,BatchCRF
from .customize_layer import SeqAnnLoss,SeqAnnModel
from .callback import CategoricalMetric,TensorboardCallback,TensorboardWriter
from .callback import EarlyStop,GFFCompare,SeqFigCallback
from .executer import ModelExecutor
from ..process.pipeline import Pipeline 
from ..process.data_processor import AnnSeqProcessor
from ..utils.utils import split,get_subdict
from ..genome_handler.utils import get_subseqs,ann_count
from ..genome_handler.region_extractor import GeneInfoExtractor
from ..data_handler.seq_converter import SeqConverter
from ..genome_handler import ann_seq_processor
from ..process.data_generator import SeqGenerator

class SeqAnnFacade:
    def __init__(self):
        self._settings={}
        self.train_seqs = None
        self.val_seqs = None
        self.test_seqs = None
        self.train_ann_seqs = None
        self.val_ann_seqs = None
        self.test_ann_seqs = None
        self._extra_train_callbacks = []
        self._extra_val_callbacks = []
        self._extra_other_callbacks = []
        self._extra_test_callbacks = []
        self._model_executor = ModelExecutor()
        self._path = None

    def clean_callbacks(self):
        self._extra_train_callbacks = []
        self._extra_val_callbacks = []
        self._extra_other_callbacks = []
        self._extra_test_callbacks = []
 
    def set_optimizer(self,loss=None,optimizer_settings=None,
                     grad_clip=None,grad_norm=None,optimizer_class=None):
        param=locals()
        del param['self']
        self._settings['optimizer'] = param
        self._model_executor = ModelExecutor()
        for key,value in param.items():
            if hasattr(self._model_executor,key):
                if value is not None:
                    setattr(self._model_executor,key,value)
            else:
                raise Exception(str(self._model_executor)+" has no attribute "+key)
        return self

    @property
    def ann_types(self):
        if self.train_ann_seqs is not None:
            return self.train_ann_seqs.ANN_TYPES
        elif self.test_ann_seqs is not None:
            return self.test_ann_seqs.ANN_TYPES
        else:
            raise Exception("Object has not set train_ann_seqs or test_ann_seqs yet.")

    def set_data(self,seqs,ann_seqs,ratio=None):
        ratio = ratio or [0.7,0.2,0.1]
        self._settings['data']={'ratio':ratio}
        ids = list(seqs.keys())
        random.shuffle(ids)
        sub_ids = split(ids,ratio)
        self.train_seqs = get_subdict(sub_ids[0],seqs)
        self.train_ann_seqs = get_subseqs(sub_ids[0],ann_seqs)
        self.val_seqs = get_subdict(sub_ids[1],seqs)
        self.val_ann_seqs = get_subseqs(sub_ids[1],ann_seqs)
        self.test_seqs = get_subdict(sub_ids[2],seqs)
        self.test_ann_seqs = get_subseqs(sub_ids[2],ann_seqs)
        ann_seqs = [self.train_ann_seqs,self.val_ann_seqs,self.test_ann_seqs]
        self._validate_ann_seqs(ann_seqs)

    def _validate_ann_seqs(self,ann_seqs):
        for ann_seq in ann_seqs:
            count = ann_count(ann_seq)
            print(len(ann_seq),count)
            count = np.array([int(val) for val in count.values()])
            if (count==0).any():
                raise Exception("There are some annotations missing is the dataset")

    def add_early_stop(self,target=None,optimize_min=None,patient=None,
                       saved_best_weights=None,restored_best_weights=None):
        param = locals()
        del param['self']
        early_stop = EarlyStop()
        self._settings['early_stop'] = param
        for key,value in param.item():
            if hasattr(early_stop,key):
                if value is not None:
                    setattr(early_stop,key,vlue)
            else:
                raise Exception(str(early_stop)+" has no attribute "+key)
        self._extra_other_callbacks.append(early_stop)
        return self

    def add_seq_fig(self,seq,ann_seq,colors_settings=None,prefix=None):
        if self._writer is None:
            raise Exception("Writer must be set first")
        seq = SeqConverter().seq2vecs(seq)
        seq = np.transpose(np.array([seq]),[0,2,1])
        seq = torch.from_numpy(seq).type('torch.FloatTensor').cuda()
        ann_seq = ann_seq_processor.seq2vecs(ann_seq)
        colors_settings = colors_settings or {'other':'blue','exon':'red','intron':'yellow'}
        colors=[colors_settings[type_]for type_ in self.ann_types]
        seq_fig = SeqFigCallback(self._writer,seq,ann_seq)
        seq_fig.class_names=self.ann_types
        seq_fig.colors=colors
        seq_fig.prefix=prefix
        self._extra_other_callbacks.append(seq_fig)
        return self

    def set_path(self,path):
        self._path = path
        self._writer = TensorboardWriter(SummaryWriter(path))
    
    def _create_gff_compare(self,path):
        simplify_map = {'gene':['exon','intron'],'other':['other']}
        gff_compare = GFFCompare(self.ann_types,path,simplify_map=simplify_map)
        return gff_compare

    def _create_categorical_metric(self,prefix=None):
        metric = CategoricalMetric()
        metric.class_num=len(self.ann_types)
        metric.class_names=self.ann_types
        metric.prefix=prefix
        return metric

    def _create_default_train_callbacks(self,path=None):
        train_callbacks = []
        val_callbacks = []
        train_metric = self._create_categorical_metric()
        val_metric = self._create_categorical_metric(prefix='val')  
        train_callbacks.append(train_metric)
        val_callbacks.append(val_metric)
        if path is not None:
            val_callbacks.append(self._create_gff_compare(path))
        return train_callbacks,val_callbacks

    def _create_default_test_callbacks(self,path=None):
        callbacks = []
        test_metric = self._create_categorical_metric(prefix='test')  
        callbacks.append(test_metric)
        if path is not None:
            callbacks.append(self._create_gff_compare(path))
        return callbacks

    def train(self,model,epoch_num=100,batch_size=32):
        train_callbacks = list(self._extra_train_callbacks)
        val_callbacks = list(self._extra_val_callbacks)
        other_callbacks = list(self._extra_other_callbacks)
        train_callbacks_, val_callbacks_ = self._create_default_train_callbacks(self._path)
        train_callbacks += train_callbacks_
        val_callbacks += val_callbacks_
        self._validate_ann_seqs([self.train_ann_seqs,self.val_ann_seqs])
        tensorboard = None
        if self._writer is not None:
            tensorboard = TensorboardCallback(self._writer)
            other_callbacks += [tensorboard]
            settings = {'epoch_num':epoch_num,'batch_size':batch_size}
            settings.update(self._settings)
            with open(self._path+"/facade_train_setting.json",'w') as fp:
                fp.write(str(settings))
        train_gen = SeqGenerator()
        val_gen = SeqGenerator()
        train_gen.batch_size=val_gen.batch_size=batch_size
        train_gen.return_extra_info=val_gen.return_extra_info=True
        train_gen.order=val_gen.order='NCL'
        train_gen.order_target=val_gen.order_target=['answers','inputs']
        train_gen.pad_value=val_gen.pad_value={'answers':-1,'inputs':0}
        worker = TrainWorker(train_gen,val_gen,self._model_executor)
        worker.epoch_num=epoch_num
        worker.train_callbacks=train_callbacks
        worker.val_callbacks=val_callbacks
        worker.other_callbacks=other_callbacks
        worker.writer=self._writer
        data_ = {'training':{'inputs':self.train_seqs,'answers':self.train_ann_seqs},
                 'validation':{'inputs':self.val_seqs,'answers':self.val_ann_seqs}
        data = AnnSeqData(data_,discard_invalid_seq=True).process()
        pipeline = Pipeline(model,data,worker)
        pipeline.path=self._path
        pipeline.execute()
        record = pipeline.result
        return record

    def test(self,model,batch_size=32):
        self._validate_ann_seqs([self.test_ann_seqs])
        callbacks = self._create_default_test_callbacks(self._path)
        if self._writer is not None:
            tensorboard = self.TensorboardCallback(self._writer)
            callbacks.append(gff_compare)
            settings={'batch_size':batch_size}
            settings.update(self._settings)
            with open(self._path+"/facade_test_setting.json",'w') as fp:
                fp.write(str(settings))
        gen = SeqGenerator()
        gen.batch_size=batch_size
        gen.return_extra_info=True
        gen.order='NCL'
        gen.order_target=['answers','inputs']
        gen.pad_value={'answers':-1,'inputs':0}
        worker = TestWorker(gen,self._model_executor)
        worker.callbacks=callbacks
        worker.writer=writer
        data = AnnSeqData({'testing':{'inputs':self.test_seqs,'answers':self.test_ann_seqs}},
                          discard_invalid_seq=True).process()
        pipeline = Pipeline(model,data,worker)
        pipeline.path=self._path
        pipeline.execute()
        record = pipeline.result
        return record
