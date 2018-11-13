from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer,SeqInfoContainer
from sequence_annotation.genome_handler.ann_seq_processor import get_background,get_seq_with_added_type,get_one_hot,simplify_seq

def get_genome_region_info(ann_genome,focus_types=None):
    """Get region information about genome"""
    genome_region_info = SeqInfoContainer()
    extractor = RegionExtractor()
    for ann_seq in ann_genome:
        genome_region_info.add(extractor.extract(ann_seq,focus_types))
    return genome_region_info

def get_backgrounded_genome(ann_genome,frontground_types,background_type):
    """Make genome with background annotation"""
    backgrounded_genome = AnnSeqContainer()
    backgrounded_genome.ANN_TYPES = set(ann_genome.ANN_TYPES + [background_type])
    for ann_seq in ann_genome:
        background = get_background(ann_seq,frontground_types=frontground_types)
        temp = get_seq_with_added_type(ann_seq,{background_type:background}) 
        backgrounded_genome.add(temp)
    return backgrounded_genome

def get_one_hot_genome(ann_genome,method='max',focus_types=None):
    """Make genome into one-hot encoded"""
    one_hot_genome = AnnSeqContainer()
    one_hot_genome.ANN_TYPES = ann_genome.ANN_TYPES
    for ann_seq in ann_genome:
        one_hot_item = get_one_hot(ann_seq,method=method,focus_types=focus_types)
        one_hot_genome.add(one_hot_item)
    return one_hot_genome

def simplify_genome(ann_genome,replace):
    simplied_genome = AnnSeqContainer()
    simplied_genome.ANN_TYPES = list(replace.keys())
    for ann_seq in ann_genome:
        simplified_seq = simplify_seq(ann_seq,replace)
        simplied_genome.add(simplified_seq)
    return simplied_genome