import deepdish as dd
from ..utils.utils import read_fasta,BASIC_GENE_ANN_TYPES
from .seq_container import AnnSeqContainer
from .ann_genome_processor import get_mixed_genome,simplify_genome,is_one_hot_genome

def select_data_by_length(fasta,ann_seqs,min_len=None,max_len=None,ratio=None):
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

def select_data_by_length_each_type(fasta,ann_seqs,**kwargs):
    if len(set(BASIC_GENE_ANN_TYPES) - set(ann_seqs.ANN_TYPES)) > 0:
        raise Exception("ANN_TYPES should include {}, but "
                        "got {}".format(BASIC_GENE_ANN_TYPES,ann_seqs.ANN_TYPES))
    multiple_exon_transcripts = []
    single_exon_transcripts = []
    no_transcripts = []
    #Classify sequence
    for ann_seq in ann_seqs:
        #If it is multiple exon transcript
        if sum(ann_seq.get_ann('intron')) > 0:
            multiple_exon_transcripts.append(ann_seq)
        #If it is single exon transcript
        elif sum(ann_seq.get_ann('exon')) > 0:
            single_exon_transcripts.append(ann_seq)
        #If there is no transcript
        else:
            no_transcripts.append(ann_seq)
            
    selected_seqs = {}
    selected_anns = ann_seqs.copy()
    selected_anns.clean()
    
    for seqs in [multiple_exon_transcripts,single_exon_transcripts,no_transcripts]:
        median_length = median([seq.length for seq in seqs])
        for seq in seqs:
            if seq.length <= median_length:
                selected_seqs[seq.id] = fasta[seq.id]
                selected_anns.add(seq)
    return selected_seqs,selected_anns

def _preprocess(ann_seqs,before_mix_simplify_map=None,simplify_map=None):
    if before_mix_simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs,before_mix_simplify_map)
    ann_seqs = get_mixed_genome(ann_seqs)
    if simplify_map is not None:
        ann_seqs = simplify_genome(ann_seqs,simplify_map)
    if not is_one_hot_genome(ann_seqs):
        raise Exception("Genome is not one-hot encoded")
    return ann_seqs

def select_data(fasta_path,ann_seqs_path,chroms_list,before_mix_simplify_map=None,
                simplify_map=None,gene_map=None,select_func=None,
                select_each_type=False,codes=None,**kwargs):

    if select_func is None:
        if select_each_type:
            select_func = select_data_by_length_each_type
        else:
            select_func = select_data_by_length
        
    if codes is not None:
        codes = set(list(codes.upper()))
        
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
                    seq = fasta[ann_seq.id]
                    add_seq=True
                    if codes is not None:
                        if len(set(list(seq.upper()))-codes) > 0:
                            add_seq=False
                            print("Discard sequence, {}, due to dirty codes in it".format(ann_seq.id))
                    if add_seq:
                        selected_anns.add(ann_seq)
                        selected_seqs[ann_seq.id] = seq
            selected_seqs,selected_anns = select_func(selected_seqs,selected_anns,**kwargs)
            selected_anns = _preprocess(selected_anns,before_mix_simplify_map,simplify_map)
            data_ = selected_seqs,selected_anns.to_dict()
        data.append(data_)
    return data
    