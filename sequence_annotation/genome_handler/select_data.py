import deepdish as dd
from ..utils.seq_converter import DNA_CODES
from ..utils.utils import read_fasta, BASIC_GENE_ANN_TYPES
from .seq_container import AnnSeqContainer
from .ann_genome_processor import get_mixed_genome, simplify_genome, is_one_hot_genome

def select_data_by_length(fasta, ann_seqs, min_len=None,
                          max_len=None, ratio=None):
    seq_lens = sorted([len(seq) for seq in ann_seqs])
    ratio = ratio or 1
    min_len = min_len or 0
    max_len = max_len or max(seq_lens)
    selected_lens = []
    for length in seq_lens:
        if min_len <= length <= max_len:
            selected_lens.append(length)
    max_len = selected_lens[:int(round(ratio * len(selected_lens)))][-1]
    selected_fasta = {}
    selected_anns = ann_seqs.copy()
    selected_anns.clean()
    for seq in ann_seqs:
        if min_len <= len(seq) <= max_len:
            selected_fasta[seq.id] = fasta[seq.id]
            selected_anns.add(ann_seqs.get(seq.id))
    print(
        "Total number is {}, selected number is {}, max length is {}".format(
            len(fasta),
            len(selected_fasta),
            max_len))
    return selected_fasta, selected_anns

def classify_ann_seqs(ann_seqs):
    selected_anns = ann_seqs.copy()
    selected_anns.clean()
    multiple_exon_region_anns = selected_anns.copy()
    single_exon_region_anns = selected_anns.copy()
    no_exon_region_anns = selected_anns.copy()
    for ann_seq in ann_seqs:
        # If it is multiple exon
        if sum(ann_seq.get_ann('intron')) > 0:
            multiple_exon_region_anns.add(ann_seq)
        # If it is single exon
        elif sum(ann_seq.get_ann('exon')) > 0:
            single_exon_region_anns.add(ann_seq)
        # If there is no exon 
        else:
            no_exon_region_anns.add(ann_seq)
    data = {}
    data['multiple_exon_region'] = multiple_exon_region_anns
    data['single_exon_region'] = single_exon_region_anns
    data['no_exon_region'] = no_exon_region_anns
    return data

def classify(fasta,ann_seqs):
    multiple_exon_region_fasta = {}
    single_exon_region_fasta = {}
    no_exon_region_fasta = {}
    classified_ann_seqs = classify_ann_seqs(ann_seqs)
    multiple_exon_region_anns = classified_ann_seqs['multiple_exon_region']
    single_exon_region_anns = classified_ann_seqs['single_exon_region']
    no_exon_region_anns = classified_ann_seqs['no_exon_region']
    for ann_seq in multiple_exon_region_anns:
        multiple_exon_region_fasta[ann_seq.id] = fasta[ann_seq.id]
        
    for ann_seq in single_exon_region_anns:
        single_exon_region_fasta[ann_seq.id] = fasta[ann_seq.id]
        
    for ann_seq in no_exon_region_anns:
        no_exon_region_fasta[ann_seq.id] = fasta[ann_seq.id]

    data = {}
    data['multiple_exon_region'] = {'fasta':multiple_exon_region_fasta,
                                    'ann_seqs':multiple_exon_region_anns}

    data['single_exon_region'] = {'fasta':single_exon_region_fasta,
                                  'ann_seqs':single_exon_region_anns}

    data['no_exon_region'] = {'fasta':no_exon_region_fasta,
                              'ann_seqs':no_exon_region_anns}
    return data
            
def select_data_by_length_each_type(fasta, ann_seqs,min_len=None,
                                    max_len=None, ratio=None):
    if len(set(BASIC_GENE_ANN_TYPES) - set(ann_seqs.ANN_TYPES)) > 0:
        raise Exception(
            "ANN_TYPES should include {}, but got {}".format(
                BASIC_GENE_ANN_TYPES,
                ann_seqs.ANN_TYPES))

    selected_fasta = {}
    selected_anns = ann_seqs.copy()
    selected_anns.clean()
    data = classify(fasta,ann_seqs)
    fasta_list = [data['multiple_exon_region']['fasta'],
                  data['single_exon_region']['fasta'],
                  data['no_exon_region']['fasta']]
    ann_list = [data['multiple_exon_region']['ann_seqs'],
                data['single_exon_region']['ann_seqs'],
                data['no_exon_region']['ann_seqs']]

    for subfasta, sub_ann_seqs in zip(fasta_list, ann_list):
        data = select_data_by_length(
            subfasta,
            sub_ann_seqs,
            min_len=min_len,
            max_len=max_len,
            ratio=ratio)
        selected_fasta.update(data[0])
        selected_anns.add(data[1])

    return selected_fasta, selected_anns


def _preprocess(ann_seqs, before_mix_simplify_map=None, simplify_map=None):
    if before_mix_simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs, before_mix_simplify_map)
    ann_seqs = get_mixed_genome(ann_seqs)
    if simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs, simplify_map)
    if not is_one_hot_genome(ann_seqs):
        raise Exception("Genome is not one-hot encoded")
    return ann_seqs


def select_data(fasta_path, ann_seqs_path, chroms, before_mix_simplify_map=None,
                simplify_map=None, select_func=None,
                select_each_type=False, codes=None, **kwargs):
    codes = codes or DNA_CODES
    if select_func is None:
        if select_each_type:
            select_func = select_data_by_length_each_type
        else:
            select_func = select_data_by_length

    h5 = dd.io.load(ann_seqs_path)
    fasta = read_fasta(fasta_path)
    ann_seqs = AnnSeqContainer().from_dict(h5)
    data = None
    if len(chroms) > 0:
        selected_anns = AnnSeqContainer(ann_seqs.ANN_TYPES)
        selected_seqs = {}
        for ann_seq in ann_seqs:
            if ann_seq.chromosome_id in chroms:
                seq = fasta[ann_seq.id]
                add_seq = True
                if codes is not None:
                    if len(set(list(seq.upper())) - codes) > 0:
                        add_seq = False
                        print(
                            "Discard sequence, {}, due to dirty codes in it".format(
                                ann_seq.id))
                if add_seq:
                    selected_anns.add(ann_seq)
                    selected_seqs[ann_seq.id] = seq
        selected_seqs, selected_anns = select_func(
            selected_seqs, selected_anns, **kwargs)
        selected_anns = _preprocess(
            selected_anns,
            before_mix_simplify_map,
            simplify_map)
        data = selected_seqs, selected_anns.to_dict()
    return data


def load_data(path):
    data = dd.io.load(path)
    data = data[0], AnnSeqContainer().from_dict(data[1])
    return data

