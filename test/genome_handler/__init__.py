import os
import sys
ucsc_file_prefix = os.path.abspath(__file__+"../../data/ucsc/")+"/"
ensembl_file_prefix = os.path.abspath(__file__+"../../data/ensembl/")+"/"
from sequence_annotation.utils.exception import ReturnNoneException, ProcessedStatusNotSatisfied
from sequence_annotation.utils.exception import InvalidStrandType,NegativeNumberException
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_info_parser import UscuInfoParser,EnsemblInfoParser
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer,SeqInfoContainer
from sequence_annotation.genome_handler.sequence import AnnSequence,SeqInformation
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator,AnnChromCreator
from sequence_annotation.genome_handler.ann_seq_processor import AnnSeqProcessor
from sequence_annotation.genome_handler.ann_seq_converter import UscuSeqConverter
from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
from sequence_annotation.genome_handler.ann_seq_converter import AnnSeqConverter, UscuSeqConverter
from sequence_annotation.genome_handler.ann_seq_converter import EnsemblSeqConverter
from sequence_annotation.genome_handler.seq_status_detector import SeqStatusDetector
from sequence_annotation.genome_handler.exon_handler import ExonHandler 
from sequence_annotation.data_handler.fasta_converter import FastaConverter
from sequence_annotation.data_handler.seq_converter import SeqConverter

from .real_genome import RealGenome
from ..ann_seq_test_case import AnnSeqTestCase
from ..seq_info_test_case import SeqInfoTestCase