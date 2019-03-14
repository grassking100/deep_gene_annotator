from torch import nn
import torch
import random
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch.optim as optim
from .worker import Trainer
from .CRF import BatchCRFLoss,BatchCRF
from .customize_layer import SeqAnnlLoss,SeqAnnModel,GatedIndRnnCell
from .callback import CategoricalMetric,TensorboardCallback,TensorboardWriter
from .callback import EarlyStop,GFFCompare,SeqFigCallback
from .compiler import SimpleCompiler
from ..process.pipeline import Pipeline 
from ..process.ann_seq_data import AnnSeqData
from ..function.model_processor import SimpleModel
from ..utils.utils import split,get_subdict
from ..genome_handler.utils import get_subseqs,ann_count
from ..genome_handler.region_extractor import GeneInfoExtractor
from ..data_handler.seq_converter import SeqConverter
from ..genome_handler import ann_seq_processor
from ..function.data_generator import SeqGenerator

class ASModelSpaceFinder:
    def __init__(self,selected_fasta,selected_seqs,root,ratios=None
                 ,outlier_fasta=None,outlier_seq=None,rnn_cell=None,
                 batch_size=32,lr=1e-4,grad_clip=None,grad_norm=None,
                 affect_length_max=1000):
        self.ann_types = list(selected_seqs.ANN_TYPES)
        self.selected_fasta = {}
        self.rnn_cell = rnn_cell or GatedIndRnnCell
        self.selected_seqs = {}
        self.batch_size = batch_size
        self.lr=lr
        self.grad_clip = grad_clip
        self.grad_norm = grad_norm
        self.affect_length_max = affect_length_max
        ids = list(selected_fasta.keys())
        random.shuffle(ids)
        ratios = ratios or [0.7,0.2,0.1]
        ids = split(ids,ratios)
        for type_,sub_ids in zip(['train','val','test'],ids):
            self.selected_fasta[type_] = get_subdict(sub_ids,selected_fasta)
            self.selected_seqs[type_] = get_subseqs(sub_ids,selected_seqs)
            annotation_counter = ann_count(self.selected_seqs[type_])
            print(len(self.selected_fasta[type_]),annotation_counter)
            annotation_counter = np.array([int(val) for val in annotation_counter.values()])
            if (annotation_counter==0).any():
                raise Exception("There are some annotations missing is the "+type_+" dataset")
        self.outlier_fasta = None
        self.outlier_seq = None
        if outlier_fasta is not None and outlier_fasta is not None:
            outlier_fasta = SeqConverter().seq2vecs(outlier_fasta)
            outlier_fasta = np.transpose(np.array([outlier_fasta]),[0,2,1])
            self.outlier_fasta = torch.from_numpy(outlier_fasta).type('torch.FloatTensor').cuda()
            outlier_seq = ann_seq_processor.seq2vecs(outlier_seq)
            self.outlier_seq = outlier_seq
        self.train_counter = 0
        self.root = root
        self.space_result = {}
        self.records = {}
        self.class_num=len(selected_seqs.ANN_TYPES)
        self.data = AnnSeqData({'training':{'inputs':self.selected_fasta['train'],
                                       'answers':self.selected_seqs['train']},
                           'validation':{'inputs':self.selected_fasta['val'],
                                         'answers':self.selected_seqs['val']}
                          },discard_invalid_seq=True)
    def objective(self,space):
        self.train_counter +=1
        print(self.train_counter,space)
        for key,val in space.items():
            if key is not 'reduce_cnn_ratio':
                space[key]=int(val)
        space['rnn_outputs']=[space['rnn_output']]*space['rnn_num']
        space['cnn_outputs']=[space['cnn_output']]*space['cnn_num']
        space['cnn_kernel_sizes']=[space['cnn_kernel_size']]*space['cnn_num']
        if space['cnn_num']*space['cnn_kernel_size']>=self.affect_length_max:
            raise Exception("CNN with such large number and size will cause sequence not to be complete")
        model = SeqAnnModel(in_channels=4,out_channels=len(self.ann_types),
                            rnn_cell_class=self.rnn_cell,
                            init_value=1,
                            with_pwm=space['PWM'],
                            use_CRF=True,rnn_layer_norm=True,**space).cuda()
        id_ = 'model_'+str(self.train_counter)
        record = self.train(id_,model)
        best = max(record['val_macro_F1'])
        self.space_result[self.train_counter] = {'space':space,'val_macro_F1':best}
        self.records[self.train_counter] = record
        with open(self.root+'/'+id_+"/record.txt","w") as fp:
            fp.write("Space:\n")
            fp.write(str(space))
            fp.write("\nBest validation macro F1:")
            fp.write(str(best))
        pd.DataFrame.from_dict(record).to_csv(self.root+'/'+id_+"/history.csv",index=None)
        return {'loss':-best,'status': STATUS_OK,'eval_time': time.time()}
    def train(self,id_,model):
        ANN_TYPES = self.selected_seqs['train'].ANN_TYPES
        loss = BatchCRFLoss(model.CRF.transitions).cuda()
        compiler = SimpleCompiler(lambda params:optim.Adam(params,lr=self.lr),loss,
                                  grad_clip=self.grad_clip,grad_norm=self.grad_norm)
        early_stop = EarlyStop(target='val_macro_F1',optimized="max",patient=16)
        writer = TensorboardWriter(SummaryWriter(self.root+"/"+id_))
        gff_compare = GFFCompare(ANN_TYPES,self.root+"/"+id_,
                                 simplify_map={'gene':['exon','intron'],'other':['other']})
        builder = SimpleModel(model)

        train_metric = CategoricalMetric(class_num=self.class_num,ignore_index=-1,
                                         class_names=self.ann_types,BatchCRF=model.CRF)
        val_metric = CategoricalMetric(prefix='val',class_num=self.class_num,ignore_index=-1,
                                       class_names=self.ann_types,BatchCRF=model.CRF)
        tensorboard = TensorboardCallback(writer)
        colors = {'other':'blue','exon':'red','intron':'yellow'}
        val_callbacks=[val_metric,early_stop,gff_compare]
        if self.outlier_fasta is not None:
            seq_fig = SeqFigCallback(writer,self.outlier_fasta,self.outlier_seq,
                                     class_names=self.ann_types,
                                     colors=[colors[type_]for type_ in self.ann_types],prefix='test')
            val_callbacks.append(seq_fig)
        worker = Trainer(batch_size=self.batch_size,return_extra_info=True, order='NCL',
                         order_target=['answers','inputs'],pad_value={'answers':-1,'inputs':0},
                         epoch_num=100,generator=SeqGenerator,
                         train_callbacks=[train_metric],val_callbacks=val_callbacks,
                         other_callbacks=[tensorboard],
                        writer=writer)
        pipeline = Pipeline(builder,self.data,worker,compiler,
                            is_prompt_visible=True)
        pipeline.execute()
        record = pipeline._worker.result.data
        return record