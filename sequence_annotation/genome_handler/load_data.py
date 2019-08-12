import deepdish as dd
from ..utils.fasta import read_fasta
from .seq_container import AnnSeqContainer
from .utils import select_seq
from .ann_genome_processor import get_mixed_genome,simplify_genome

def _preprocess(ann_seqs,before_mix_simplify_map=None,simplify_map=None):
    if before_mix_simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs,before_mix_simplify_map)
    ann_seqs = get_mixed_genome(ann_seqs)
    if simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs,simplify_map)
    return ann_seqs

def _load_data(fasta,ann_seqs,chroms,max_len=None,simplify_map=None,
               before_mix_simplify_map=None,handle_outlier=False):
    selected_ann_seqs = AnnSeqContainer()
    selected_ann_seqs.ANN_TYPES = ann_seqs.ANN_TYPES
    selected_seqs = {}
    for ann_seq in ann_seqs:
        if ann_seq.chromosome_id in chroms:
            selected_ann_seqs.add(ann_seq)
            selected_seqs[ann_seq.id] = fasta[ann_seq.id]

    data = select_seq(selected_seqs,selected_ann_seqs,max_len=max_len)
    selected_seqs,selected_ann_seqs,outlier_seq,outlier_ann = data
    selected_ann_seqs = _preprocess(selected_ann_seqs,before_mix_simplify_map,simplify_map)

    if handle_outlier:
        outlier = AnnSeqContainer()
        outlier.ANN_TYPES = outlier_ann.ANN_TYPES
        outlier.add(outlier_ann)
        outlier = _preprocess(outlier,before_mix_simplify_map,simplify_map)
        outlier_ann = outlier.data[0]
        data = selected_seqs,selected_ann_seqs,outlier_seq,outlier_ann
    else:
        data = selected_seqs,selected_ann_seqs

    return data

def load_data(fasta_path,ann_seqs_path,train_chroms,val_chroms=None,test_chroms=None,max_len=None,
              simplify_map=None,before_mix_simplify_map=None,handle_outlier=None):
    _handle_outlier = handle_outlier or [False,True,False]
    h5=dd.io.load(ann_seqs_path)
    fasta = read_fasta(fasta_path)
    ann_seqs = AnnSeqContainer().from_dict(h5)
    train_data = _load_data(fasta,ann_seqs,train_chroms,max_len,
                            simplify_map,before_mix_simplify_map,
                            handle_outlier=_handle_outlier[0])
    val_data = None
    test_data = None
    if val_chroms is not None:
        val_data = _load_data(fasta,ann_seqs,val_chroms,max_len,
                              simplify_map,before_mix_simplify_map,
                              handle_outlier=_handle_outlier[1])
    if test_chroms is not None:
        test_data = _load_data(fasta,ann_seqs,test_chroms,max_len,
                               simplify_map,before_mix_simplify_map,
                               handle_outlier=_handle_outlier[2])
    return train_data,val_data,test_data
    