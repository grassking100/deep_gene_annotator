import deepdish as dd
from ..utils.utils import read_fasta
from .seq_container import AnnSeqContainer
from .ann_genome_processor import get_mixed_genome,simplify_genome,get_genome_with_site_ann

def select_data_by_length(fasta,ann_seqs,min_len=None,max_len=None,ratio=None,**kwargs):
    seq_lens = [len(seq) for seq in ann_seqs]
    seq_lens.sort()
    ratio = ratio or 1
    min_len = min_len or 0
    max_len = max_len or max(seq_lens)
    selected_lens = []
    for length in seq_lens:
        if min_len <= length <= max_len:
            selected_lens.append(length)
    max_len = selected_lens[:int(ratio*len(selected_lens))][-1]
    selected_fasta = {}
    selected_anns = AnnSeqContainer(ann_seqs.ANN_TYPES)
    for seq in ann_seqs:
        if min_len <= len(seq) <= max_len:
            selected_fasta[seq.id]=fasta[seq.id]
            selected_anns.add(ann_seqs.get(seq.id))
    return selected_fasta,selected_anns

def _preprocess(ann_seqs,before_mix_simplify_map=None,simplify_map=None):
    if before_mix_simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs,before_mix_simplify_map)
    ann_seqs = get_mixed_genome(ann_seqs)
    if simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs,simplify_map)
    return ann_seqs

def load_data(fasta_path,ann_seqs_path,chroms_list,before_mix_simplify_map=None,
              simplify_map=None,site_ann_method=None,gene_map=None,
              select_func=None,**kwargs):
    if select_func is None:
        select_func = select_data_by_length
    h5=dd.io.load(ann_seqs_path)
    fasta = read_fasta(fasta_path)
    ann_seqs = AnnSeqContainer().from_dict(h5)
    data = []
    for chroms in chroms_list:
        data_ = None
        if len(chroms) > 0:
            selected_anns = AnnSeqContainer(ann_seqs.ANN_TYPES)
            selected_seqs = {}
            for ann_seq in ann_seqs:
                if ann_seq.chromosome_id in chroms:
                    selected_anns.add(ann_seq)
                    selected_seqs[ann_seq.id] = fasta[ann_seq.id]
            selected_seqs,selected_anns = select_func(selected_seqs,selected_anns,**kwargs)
            selected_anns = _preprocess(selected_anns,before_mix_simplify_map,simplify_map)
            if site_ann_method is not None:
                set_non_site = site_ann_method == 'all'
                selected_anns = get_genome_with_site_ann(selected_anns,gene_map=gene_map,set_non_site=set_non_site)
            data_ = selected_seqs,selected_anns
        data.append(data_)
    return data
    