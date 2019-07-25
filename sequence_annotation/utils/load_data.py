import deepdish as dd
from ..genome_handler.seq_container import AnnSeqContainer
from ..data_handler.fasta import read_fasta
from ..genome_handler.utils import select_seq
from ..genome_handler.ann_genome_processor import get_mixed_genome,simplify_genome

def _prprocess(ann_seqs,before_mix_simplify_map,simplify_map):
    if before_mix_simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs,before_mix_simplify_map)
    ann_seqs = get_mixed_genome(ann_seqs)
    ann_seqs = simplify_genome(ann_seqs,simplify_map)
    return ann_seqs

def _load_data(fasta,ann_seqs,max_len,chroms,simplify_map,before_mix_simplify_map=None,handle_outlier=False):
    selected_ann_seqs = AnnSeqContainer()
    selected_ann_seqs.ANN_TYPES = ann_seqs.ANN_TYPES
    selected_seqs = {}
    for ann_seq in ann_seqs:
        if int(ann_seq.chromosome_id) in chroms:
            selected_ann_seqs.add(ann_seq)
            selected_seqs[ann_seq.id] = fasta[ann_seq.id]

    data = select_seq(selected_seqs,selected_ann_seqs,max_len=max_len)
    selected_seqs,selected_ann_seqs,outlier_seq,outlier_ann = data
    selected_ann_seqs = _prprocess(selected_ann_seqs,before_mix_simplify_map,simplify_map)

    if handle_outlier:
        outlier = AnnSeqContainer()
        outlier.ANN_TYPES = outlier_ann.ANN_TYPES
        outlier.add(outlier_ann)
        outlier = _prprocess(outlier,before_mix_simplify_map,simplify_map)
        outlier_ann = outlier.data[0]
        data = selected_seqs,selected_ann_seqs,outlier_seq,outlier_ann
    else:
        data = selected_seqs,selected_ann_seqs

    return data

def load_data(fasta_path,ann_seqs_path,max_len,train_chroms,val_chroms,test_chroms,
              simplify_map,before_mix_simplify_map=None):
    h5=dd.io.load(ann_seqs_path)
    fasta = read_fasta(fasta_path)
    ann_seqs = AnnSeqContainer().from_dict(h5)
    train_data = _load_data(fasta,ann_seqs,max_len,train_chroms,
                            simplify_map,before_mix_simplify_map)
    val_data = _load_data(fasta,ann_seqs,max_len,val_chroms,
                          simplify_map,before_mix_simplify_map,handle_outlier=True)
    test_data = _load_data(fasta,ann_seqs,max_len,test_chroms,
                           simplify_map,before_mix_simplify_map)
    return train_data,val_data,test_data