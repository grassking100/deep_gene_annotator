import json
import math
import unittest
import deepdish
import numpy as np
import pandas as pd
import math
from time import gmtime, strftime
from os.path import abspath, expanduser
from sequence_annotation.genome_handler.seq_info_parser import UscuInfoParser,EnsemblInfoParser
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.exon_handler import ExonHandler
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer,SeqInfoContainer
from sequence_annotation.genome_handler.ann_seq_processor import get_background,get_seq_with_added_type,get_one_hot
from sequence_annotation.genome_handler.ann_seq_converter import EnsemblSeqConverter,UscuSeqConverter
from sequence_annotation.genome_handler.seq_status_detector import SeqStatusDetector
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator,AnnChromCreator
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
from sequence_annotation.data_handler.seq_converter import SeqConverter
from sequence_annotation.genome_handler.sequence import SeqInformation

class TestGenomeHandlePipeline(unittest.TestCase):
    def test_runable(self):
        try:
            id_ = 0
            length = 180
            #Create object to use
            non_conflict_type = ['cds','utr_5','utr_3','intron','other']
            ann_seq_container = AnnSeqContainer()
            info_container = SeqInfoContainer()
            ann_seq_container.ANN_TYPES = non_conflict_type + ['exon']
            gene_converter = EnsemblSeqConverter(extra_types=['exon'])
            ann_genome_creator = AnnGenomeCreator()
            converted_data = AnnSeqContainer()
            exon_handler = ExonHandler()
            ensembl_path = abspath(expanduser(__file__+'/../data/test_data/test_ensembl.tsv'))
            seq_info = {'chromosome':{'chr1':240},'source':'test'}
            #Read data
            data = pd.read_csv(ensembl_path,sep='\t').to_dict('record')
            #Create Annotated Genoome
            converted_data.ANN_TYPES = gene_converter.ANN_TYPES
            for seq in EnsemblInfoParser().parse(data):
                converted_seq = gene_converter.convert(seq)
                converted_seq.processed_status = "one_hot"
                further_seq = exon_handler.further_division(converted_seq)
                internal_seq = exon_handler.discard_external(further_seq)
                simple_seq = exon_handler.simplify_exon_name(internal_seq)

                converted_data.add(simple_seq)
            genome=ann_genome_creator.create(converted_data,seq_info)
            #Get one strand
            for chrom in genome.data:
                #Create sequence with wanted annotation
                background = get_background(chrom)
                complete = get_seq_with_added_type(chrom,{'other':background})
                one_hot = get_one_hot(complete,non_conflict_type,method='order')
                one_hot.add_ann('other',get_background(one_hot))
                ann_seq_container.add(one_hot)
                #Create regions for extraction
                for index in range(math.ceil(simple_seq.length/length)):
                    info = SeqInformation()
                    info.id = id_
                    id_+=1
                    info.chromosome_id = chrom.chromosome_id
                    info.strand = chrom.strand
                    info.start = index*length
                    info.end = (index+1)*length -1
                    if info.end >= simple_seq.length:
                        info.end = simple_seq.length - 1
                        info.start = info.end - length + 1
                    info_container.add(info)
            #Extract sequence    
            extracted = AnnSeqExtractor().extract(ann_seq_container,info_container)
            #Convert to storing format
            answer = {}
            for item in extracted.to_dict()['data']:
                data = {}
                for type_ in non_conflict_type:
                    data[type_] = item['data'][type_]
                answer[item['id']]=data
        except Exception as exp:
            raise exp
            self.fail("There are some unexpected exception occur.")
