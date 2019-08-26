import deepdish as dd
from ..utils.fasta import read_fasta
from .seq_container import AnnSeqContainer
from .ann_genome_processor import get_mixed_genome,simplify_genome

def select_seq(fasta,ann_seqs,min_len=None,max_len=None):
    seq_lens = [len(seq) for seq in ann_seqs]
    seq_lens.sort()
    min_len = min_len or 0
    max_len = max_len or max(seq_lens)
    selected_fasta = {}
    selected_ann_seqs = AnnSeqContainer(ann_seqs.ANN_TYPES)
    for seq in ann_seqs:
        if min_len <= len(seq) <= max_len:
            selected_fasta[seq.id]=fasta[seq.id]
            selected_ann_seqs.add(ann_seqs.get(seq.id))    
    return selected_fasta,selected_ann_seqs

def _preprocess(ann_seqs,before_mix_simplify_map=None,simplify_map=None):
    if before_mix_simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs,before_mix_simplify_map)
    ann_seqs = get_mixed_genome(ann_seqs)
    if simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs,simplify_map)
    return ann_seqs

def _load_data(fasta,ann_seqs,chrom_ids,min_len=None,max_len=None,
              simplify_map=None,before_mix_simplify_map=None):
    selected_ann_seqs = AnnSeqContainer(ann_seqs.ANN_TYPES)
    selected_seqs = {}
    for ann_seq in ann_seqs:
        if ann_seq.chromosome_id in chrom_ids:
            selected_ann_seqs.add(ann_seq)
            selected_seqs[ann_seq.id] = fasta[ann_seq.id]

    data = select_seq(selected_seqs,selected_ann_seqs,min_len=min_len,max_len=max_len)
    selected_seqs,selected_ann_seqs = data
    selected_ann_seqs = _preprocess(selected_ann_seqs,before_mix_simplify_map,simplify_map)
    data = selected_seqs,selected_ann_seqs
    return data

def load_data(fasta_path,ann_seqs_path,chroms_list,
              min_len=None,max_len=None,simplify_map=None,
              before_mix_simplify_map=None):
    h5=dd.io.load(ann_seqs_path)
    fasta = read_fasta(fasta_path)
    ann_seqs = AnnSeqContainer().from_dict(h5)
    data = []
    for chroms in chroms_list:
        data_ = None
        if len(chroms) > 0:
            data_ = _load_data(fasta,ann_seqs,chroms,
                                min_len=min_len,max_len=max_len,
                                simplify_map=simplify_map,
                                before_mix_simplify_map=before_mix_simplify_map)
        data.append(data_)
    return data
    