import os
import sys
ucsc_file_prefix = os.path.abspath(__file__+"../../data/ucsc/")+"/"
ensembl_file_prefix = os.path.abspath(__file__+"../../data/ensembl/")+"/"
from sequence_annotation.utils.exception import ReturnNoneException
from sequence_annotation.utils.exception import InvalidStrandType,NegativeNumberException
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_info_parser import UscuInfoParser,EnsemblInfoParser
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer,SeqInfoContainer
from sequence_annotation.genome_handler.sequence import AnnSequence,SeqInformation
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.genome_handler.ann_seq_processor import AnnSeqProcessor
from sequence_annotation.genome_handler.ann_seq_converter import UscuSeqConverter
from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
from .real_genome import RealGenome