import warnings
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer,SeqInfoContainer
from sequence_annotation.genome_handler import ann_seq_processor

def mixed_typed_genome_generate(seqs):
    mixed_seqs = AnnSeqContainer()
    mixed_seqs.ANN_TYPES = ['exon','intron','mix','other']
    for seq in seqs:
        mixed_seqs.add(ann_seq_processor.mixed_typed_seq_generate(seq))
    return mixed_seqs

def get_genome_region_info(ann_genome,focus_types=None):
    """Get region information about genome"""
    genome_region_info = SeqInfoContainer()
    extractor = RegionExtractor()
    for ann_seq in ann_genome:
        genome_region_info.add(extractor.extract(ann_seq,focus_types))
    return genome_region_info

def get_backgrounded_genome(ann_genome,background_type,frontground_types=None):
    """Make genome with background annotation"""
    backgrounded_genome = AnnSeqContainer()
    frontground_types = frontground_types or ann_genome.ANN_TYPES
    backgrounded_genome.ANN_TYPES = set(frontground_types + [background_type])
    for ann_seq in ann_genome:
        background = ann_seq_processor.get_background(ann_seq,frontground_types=frontground_types)
        temp = ann_seq_processor.get_seq_with_added_type(ann_seq,{background_type:background}) 
        backgrounded_genome.add(temp)
    return backgrounded_genome

def get_one_hot_genome(ann_genome,method='max',focus_types=None):
    """Make genome into one-hot encoded"""
    one_hot_genome = AnnSeqContainer()
    one_hot_genome.ANN_TYPES = ann_genome.ANN_TYPES
    for ann_seq in ann_genome:
        one_hot_item = ann_seq_processor.get_one_hot(ann_seq,method=method,focus_types=focus_types)
        one_hot_genome.add(one_hot_item)
    return one_hot_genome

def simplify_genome(ann_genome,replace):
    simplied_genome = AnnSeqContainer()
    simplied_genome.ANN_TYPES = list(replace.keys())
    for ann_seq in ann_genome:
        simplified_seq = ann_seq_processor.simplify_seq(ann_seq,replace)
        simplied_genome.add(simplified_seq)
    return simplied_genome

def class_count(ann_genome):
    ann_count = {}
    ANN_TYPES = ann_genome.ANN_TYPES
    for type_ in ANN_TYPES:
        ann_count[type_] = 0
    for ann_seq in ann_genome:
        count = ann_seq_processor.class_coun(ann_seq)
        for type_ in ANN_TYPES:
            ann_count[type_] += count[type_]
    return ann_count

def genome2dict_vec(ann_genome,ann_types=None):
    warn = ("\n\n!!!\n"
            "\tDNA sequence will be rearranged from 5' to 3'.\n"
            "\tThe plus strand sequence will stay the same,"
            " but the minus strand sequence will be flipped!\n"
            "!!!\n")
    warnings.warn(warn)
    ann_types = ann_types or ann_genome.ANN_TYPES
    dict_ = {}
    for ann_seq in ann_genome:
        dict_[str(ann_seq.id)] = ann_seq_processor.seq2vecs(ann_seq,ann_types=ann_types)
    return dict_

def vecs2genome(vecs,ids,strands,ann_types):
    #vecs shape is channel,length
    if vecs.shape[0] != len(ann_types):
        raise Exception("The number of annotation type is not match with the channel number.")
    ann_genome = AnnSeqContainer()
    ann_genome.ANN_TYPES = ann_types
    for vecs_,id_,strand in zip(vecs,ids,strands):
        ann_seq = ann_seq_processor.vecs2seq(vecs_,id_,strand,ann_types)
    ann_genome.add(ann_seq)
    return ann_genome