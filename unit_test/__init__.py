import sys
sys.append("/../../")
print(":")
from sequence_annotation.utils.exception import ReturnNoneException
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.seq_info_parser import USCUParser
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from .real_genome import RealGenome