import pandas as pd
from .sequence import AnnSequence,SeqInformation
from .seq_info_parser import BedInfoParser
from .ann_seq_converter import CodingBedSeqConverter
from .seq_container import AnnSeqContainer
from .ann_genome_creator import AnnGenomeCreator
from .ann_genome_processor import get_backgrounded_genome
from .mediator import Mediator
from ..utils import read_fai


def create_ann_region(mRNA_bed12_path,fai_path,source_name):
    #Read chromosome length file
    genome_info = read_fai(fai_path)
    #Parse the bed file and convert its data to AnnSeqContainer
    parser_12 = BedInfoParser()
    converter = CodingBedSeqConverter()
    mediator = Mediator(parser_12,converter)
    ann_seqs = mediator.create(mRNA_bed12_path)
    genome_creator = AnnGenomeCreator()
    genome = genome_creator.create(ann_seqs,{'chromosome':genome_info,'source':source_name})
    backgrounded_genome = get_backgrounded_genome(genome,'other')
    return backgrounded_genome
