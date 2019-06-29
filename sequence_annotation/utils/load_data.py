import deepdish as dd
from ..genome_handler.seq_container import AnnSeqContainer
from ..data_handler.fasta import read_fasta
from ..genome_handler.utils import select_seq
from ..genome_handler.ann_genome_processor import get_mixed_types_genome,simplify_genome

def load_data(fasta_path,ann_seqs_path,max_len,train_chroms,val_chroms,test_chroms):
    h5=dd.io.load(ann_seqs_path)
    fasta = read_fasta(fasta_path)
    ann_seqs = AnnSeqContainer().from_dict(h5)
    ann_seqs = get_mixed_types_genome(ann_seqs)
    simplify_map={'exon':['cds','utr_5','utr_3','mix'],'intron':['intron'],'other':['other']}
    ann_seqs = simplify_genome(ann_seqs,simplify_map)
    train_ann_seqs,val_ann_seqs,test_ann_seqs = AnnSeqContainer(), AnnSeqContainer(), AnnSeqContainer()
    train_ann_seqs.ANN_TYPES = val_ann_seqs.ANN_TYPES = test_ann_seqs.ANN_TYPES = ann_seqs.ANN_TYPES
    train_seqs,val_seqs,test_seqs = {},{},{}
    for ann_seq in ann_seqs:
        if int(ann_seq.chromosome_id) in train_chroms:
            train_ann_seqs.add(ann_seq)
            train_seqs[ann_seq.id] = fasta[ann_seq.id]
            
        elif int(ann_seq.chromosome_id) in val_chroms:
            val_ann_seqs.add(ann_seq)
            val_seqs[ann_seq.id] = fasta[ann_seq.id]

        elif int(ann_seq.chromosome_id) in test_chroms:
            test_ann_seqs.add(ann_seq)
            test_seqs[ann_seq.id] = fasta[ann_seq.id]

    train_data = select_seq(train_seqs,train_ann_seqs,max_len=max_len)[:2]
    val_data = select_seq(val_seqs,val_ann_seqs,max_len=max_len)
    test_data = select_seq(test_seqs,test_ann_seqs,max_len=max_len)[:2]

    return train_data,val_data,test_data