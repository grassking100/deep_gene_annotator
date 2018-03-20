import os
import sys
sys.path.append(os.path.abspath(__file__+"/../../../../"))
file_prefix = os.path.abspath(__file__+"../../data/ucsc/")+"/"
from sequence_annotation.utils.exception import ReturnNoneException
from sequence_annotation.utils.exception import InvalidStrandType,NegativeNumberException
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_info_parser import USCUParser
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.test.genome_handler.real_genome import RealGenome
from .real_genome import RealGenome