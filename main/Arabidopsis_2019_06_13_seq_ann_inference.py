import os
import sys
sys.path.append("./sequence_annotation")
import torch
torch.backends.cudnn.benchmark = True
import deepdish as dd
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.data_handler.fasta import read_fasta
from sequence_annotation.genome_handler.utils import select_seq
from sequence_annotation.genome_handler.ann_genome_processor import get_mixed_types_genome,simplify_genome
from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
from sequence_annotation.pytorch.loss import SeqAnnLoss
from sequence_annotation.pytorch.executer import ModelExecutor
from sequence_annotation.pytorch.model import SeqAnnModel,seq_ann_inference
from sequence_annotation.data_handler.fasta import write_fasta
import json

def train(fasta_path,ann_seqs_path,max_len,train_chroms,val_chroms,test_chroms,space,saved_root):
    setting = locals()
    with open(saved_root+"/setting.json","w") as outfile:
        json.dump(setting, outfile)
    h5=dd.io.load(ann_seqs_path)
    fasta = read_fasta(fasta_path)
    ann_seqs = AnnSeqContainer().from_dict(h5)
    ann_seqs = get_mixed_types_genome(ann_seqs)
    simplify_map={'exon':['cds','utr_5','utr_3','mix'],'intron':['intron'],'other':['other']}
    ann_seqs = simplify_genome(ann_seqs,simplify_map)
    selected_fasta,selected_seqs,o_f,o_s = select_seq(fasta,ann_seqs,max_len=max_len)
    train_ann_seqs = AnnSeqContainer()
    val_ann_seqs = AnnSeqContainer()
    test_ann_seqs = AnnSeqContainer()
    train_ann_seqs.ANN_TYPES = val_ann_seqs.ANN_TYPES = test_ann_seqs.ANN_TYPES = selected_seqs.ANN_TYPES
    train_seqs = {}
    val_seqs = {}
    test_seqs = {}
    for ann_seq in selected_seqs:
        if int(ann_seq.chromosome_id) in train_chroms:
            train_ann_seqs.add(ann_seq)
            train_seqs[ann_seq.id] = selected_fasta[ann_seq.id]
        elif int(ann_seq.chromosome_id) in val_chroms:
            val_ann_seqs.add(ann_seq)
            val_seqs[ann_seq.id] = selected_fasta[ann_seq.id]
        elif int(ann_seq.chromosome_id) in test_chroms:
            test_ann_seqs.add(ann_seq)
            test_seqs[ann_seq.id] = selected_fasta[ann_seq.id]
    facade = SeqAnnFacade()
    facade.train_seqs,facade.train_ann_seqs = train_seqs,train_ann_seqs
    facade.val_seqs,facade.val_ann_seqs = val_seqs,val_ann_seqs
    facade.test_seqs,facade.test_ann_seqs = test_seqs,test_ann_seqs
    facade.set_path(saved_root)
    executor = ModelExecutor()
    executor.loss = SeqAnnLoss(intron_coef=1,other_coef=1,nonoverlap_coef=1)
    executor.inference = seq_ann_inference
    facade.executor = executor
    facade.simplify_map = {'gene':['exon','intron'],'other':['other']}
    facade.add_seq_fig(o_f,o_s)
    facade.add_early_stop(target='val_loss',optimize_min=True,patient=10,
                          save_best_weights=True,restore_best_weights=True)
    write_fasta(saved_root+'/train.fasta',facade.train_seqs)
    write_fasta(saved_root+'/val.fasta',facade.val_seqs)
    write_fasta(saved_root+'/test.fasta',facade.test_seqs)
    model = SeqAnnModel(**space).cuda()
    train_record = facade.train(model,batch_size=32,epoch_num=100)
    test_record = facade.test(model,batch_size=32)
    #torch.save(model.state_dict(), saved_root+"/model.pth")
if __name__ == '__main__':
    print(sys.argv)
    os.environ["CUDA_VISIBLE_DEVICES"] =  sys.argv[1]
    fasta_path,ann_seqs_path = sys.argv[2],sys.argv[3]
    max_len = int(sys.argv[4])
    train_chroms = [int(v) for v in sys.argv[5].split(",")]
    val_chroms = [int(v) for v in sys.argv[6].split(",")]
    test_chroms = [int(v) for v in sys.argv[7].split(",")]
    saved_root = sys.argv[8]
    space={
        'input_size':4,'out_channels':2,'use_sigmoid':True,
        'pwm_before_rnns':False,'last_kernel_size':1,'rnns_type':'GRU',
        'cnns_setting':{
            "ln_mode":"after_activation",
            'num_layers':4,
            'with_pwm':False,
            'out_channels':[16,16,16,16],
            'kernel_sizes':[60,60,60,60]
        },
        'rnns_setting':{
            'num_layers':4,
            'hidden_size':32,
            'batch_first':True,
            'bidirectional':True
        }
    }
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)
    script_path = sys.argv[0]
    command = 'cp -t '+saved_root+' '+script_path
    print(command)
    os.system(command)
    train(fasta_path,ann_seqs_path,max_len,train_chroms,val_chroms,test_chroms,space,saved_root)