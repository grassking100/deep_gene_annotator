import os
import torch
torch.backends.cudnn.benchmark = True
import deepdish as dd
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.data_handler.fasta import read_fasta
from sequence_annotation.genome_handler.utils import select_seq
from sequence_annotation.genome_handler.ann_genome_processor import get_mixed_types_genome,simplify_genome
from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
from sequence_annotation.pytorch.loss import SeqAnnLoss
from sequence_annotation.pytorch.model import SeqAnnModel,seq_ann_inference
from sequence_annotation.data_handler.fasta import write_fasta

if __name__ == '__main__':
    ann_seqs_path = ''
    seq_fasta_path = ''
    gpu_id = '0'
    max_len=2000
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    h5=dd.io.load('../io/Arabidopsis_thaliana/data/2019_05_11/result/selected_region.h5')
    fasta = read_fasta('../io/Arabidopsis_thaliana/data/2019_05_11/result/selected_region.fasta')
    ann_seqs = AnnSeqContainer().from_dict(h5)
    ann_seqs = get_mixed_types_genome(ann_seqs)
    simplify_map={'exon':['cds','utr_5','utr_3','mix'],'intron':['intron'],'other':['other']}
    ann_seqs = simplify_genome(ann_seqs,simplify_map)
    selected_fasta,selected_seqs,o_f,o_s = select_seq(fasta,ann_seqs,max_len=2000)
    facade = SeqAnnFacade()
    facade.assign_data_by_random(selected_fasta,selected_seqs,ratio=[0.7,0.3,0])
    facade.set_path('../io/record/arabidopsis_2019_05_15/debug')
    facade.set_optimizer(optimizer_settings={'lr':1e-3})
    facade.simplify_map = {'gene':['exon','intron'],'other':['other']}
    facade.add_seq_fig(o_f,o_s,inference=None)
    write_fasta('../io/record/arabidopsis_2019_05_22/trial_01/train.fasta',facade.train_seqs)
    write_fasta('../io/record/arabidopsis_2019_05_22/trial_01/val.fasta',facade.val_seqs)
    space={
        'cnns_setting':{
            "ln_mode":"after_activation",
            'num_layers':4,
            'with_pwm':False
            'out_channels':[16,16,16,16],
            'kernel_sizes':[60,60,60,60]
        },
        'rnns_setting':{
            'num_layers':4,
            'hidden_size':32,
            'batch_first':True,
            'bidirectional':True
        },
        'pwm_before_rnns':False,
        'rnns_type':'GRU',
        'last_kernel_size':1
    }
    model = SeqAnnModel(input_size=4,out_channels=3,use_sigmoid=True,**space).cuda()
    train_record = facade.train(model,batch_size=128,epoch_num=20)