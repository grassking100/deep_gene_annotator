from . import RegionExtractor
from . import SeqInfoContainer, AnnSeqContainer
from . import AnnSeqProcessor

class AnnGenomeProcessor():
    
    def get_genome_region_info(self,ann_genome,focus_types=None):
        """Get region information about genome"""
        genome_region_info = SeqInfoContainer()
        extractor = RegionExtractor()
        for ann_seq in ann_genome:
            genome_region_info.add(extractor.extract(ann_seq,focus_types))
        return genome_region_info

    def get_backgrounded_genome(self,ann_genome,frontground_types,background_type):
        """Make genome with background annotation"""
        ann_seq_processor = AnnSeqProcessor()
        backgrounded_genome = AnnSeqContainer()
        backgrounded_genome.ANN_TYPES = set(ann_genome.ANN_TYPES + [background_type])
        for ann_seq in ann_genome:
            background = ann_seq_processor.get_background(ann_seq,frontground_types=frontground_types)
            temp = ann_seq_processor.get_seq_with_added_type(ann_seq,{background_type:background}) 
            backgrounded_genome.add(temp)
        return backgrounded_genome

    def get_one_hot_genome(self,ann_genome,method='max',non_conflict_types=None):
        """Make genome into one-hot encoded"""
        non_conflict_types = non_conflict_types or ann_genome.ANN_TYPES
        one_hot_genome = AnnSeqContainer()
        one_hot_genome.ANN_TYPES = ann_genome.ANN_TYPES
        ann_seq_processor = AnnSeqProcessor()
        for ann_seq in ann_genome:
            one_hot_item = ann_seq_processor.get_one_hot(ann_seq,method=method,focus_types = non_conflict_types)
            one_hot_genome.add(one_hot_item)
        return one_hot_genome