import sys
from .seq_container import AnnSeqContainer
from . import ann_seq_processor


def get_backgrounded_genome(
        ann_genome, background_type, frontground_types=None):
    """Make genome with background annotation"""
    backgrounded_genome = AnnSeqContainer()
    frontground_types = frontground_types or ann_genome.ANN_TYPES
    backgrounded_genome.ANN_TYPES = set(
        ann_genome.ANN_TYPES + [background_type])
    for ann_seq in ann_genome:
        background = ann_seq_processor.get_background(
            ann_seq, frontground_types=frontground_types)
        if background_type in ann_seq.ANN_TYPES:
            backgrounded_seq = ann_seq.copy()
            backgrounded_seq.set_ann(background_type, background)
        else:
            backgrounded_seq = ann_seq_processor.get_seq_with_added_type(
                ann_seq, {background_type: background})
        backgrounded_genome.add(backgrounded_seq)
    return backgrounded_genome


def get_one_hot_genome(ann_genome, method='max', focus_types=None):
    """Make genome into one-hot encoded"""
    one_hot_genome = AnnSeqContainer(ann_genome.ANN_TYPES)
    for ann_seq in ann_genome:
        one_hot_item = ann_seq_processor.get_one_hot(
            ann_seq, method=method, focus_types=focus_types)
        one_hot_genome.add(one_hot_item)
    return one_hot_genome


def is_one_hot_genome(ann_genome, focus_types=None):
    """Is genome one-hot encoded"""
    for ann_seq in ann_genome:
        if not ann_seq_processor.is_one_hot(ann_seq, focus_types):
            return False
    return True


def get_mixed_genome(ann_genome, verbose=True):
    """Make genome into one-hot encoded"""
    map_ = {}
    max_val = 2**len(ann_genome.ANN_TYPES) - 1
    format_ = "0{}b".format(len(ann_genome.ANN_TYPES))
    for id_ in range(0, max_val + 1):
        bits = list(format(id_, format_))
        name = ""
        for bit, type_ in zip(bits, ann_genome.ANN_TYPES):
            if bit == '1':
                if name == "":
                    name = type_
                else:
                    name = name + "_" + type_
        if name == "":
            name = 'DANGER'
        map_[name] = [int(bit) for bit in bits]
    mixed_genome = AnnSeqContainer()
    for index, ann_seq in enumerate(ann_genome):
        if verbose:
            status = round(100 * index / len(ann_genome))
            print("Parsed {}% of data".format(status), end='\r')
            sys.stdout.write('\033[K')
        mixed_types_seq = ann_seq_processor.get_mixed_seq(ann_seq, map_)
        if mixed_genome.ANN_TYPES is None:
            mixed_genome.ANN_TYPES = mixed_types_seq.ANN_TYPES
        mixed_genome.add(mixed_types_seq)
    return mixed_genome


def simplify_genome(ann_genome, replace):
    simplied_genome = AnnSeqContainer(list(replace.keys()))
    for ann_seq in ann_genome:
        simplified_seq = ann_seq_processor.simplify_seq(ann_seq, replace)
        simplied_genome.add(simplified_seq)
    return simplied_genome


def class_count(ann_genome):
    ann_count = {}
    ANN_TYPES = ann_genome.ANN_TYPES
    for type_ in ANN_TYPES:
        ann_count[type_] = 0
    for ann_seq in ann_genome:
        count = ann_seq_processor.class_count(ann_seq)
        for type_ in ANN_TYPES:
            ann_count[type_] += count[type_]
    return ann_count


def genome2dict_vec(ann_genome, ann_types):
    dict_ = {}
    for ann_seq in ann_genome:
        dict_[str(ann_seq.id)] = ann_seq_processor.seq2vecs(ann_seq, ann_types)
    return dict_


def get_sub_ann_seqs(ann_seq_container, seq_info_container):
    result = AnnSeqContainer(ann_seq_container.ANN_TYPES)
    for seq_info in seq_info_container.data:
        single_strand_chrom = ann_seq_container.get(
            str(seq_info.chromosome_id) + "_" + seq_info.strand)
        seq_ann = single_strand_chrom.get_subseq(seq_info.start, seq_info.end)
        seq_ann.id = seq_info.id
        result.add(seq_ann)
    return result


def get_genome_with_site_ann(ann_genome, **kwargs):
    """Get annotaed genome with site annotation"""
    returned = None
    for ann_seq in ann_genome:
        returned_seq = ann_seq_processor.get_seq_with_site_ann(
            ann_seq, **kwargs)
        if returned is None:
            returned = AnnSeqContainer(returned_seq.ANN_TYPES)
        returned.add(returned_seq)
    return returned
