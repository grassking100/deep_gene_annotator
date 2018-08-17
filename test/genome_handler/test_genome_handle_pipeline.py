import os
import sys
import json
import math
import unittest
import deepdish
import numpy as np
import pandas as pd
import math
from time import gmtime, strftime
from os.path import abspath, expanduser
from . import UscuInfoParser, EnsemblInfoParser
from . import AnnChromCreator, AnnGenomeCreator
from . import RegionExtractor, ExonHandler
from . import SeqInfoGenerator, AnnSeqContainer
from . import SeqInfoContainer
from . import AnnSeqExtractor, AnnSeqProcessor
from . import UscuSeqConverter, EnsemblSeqConverter
from . import SeqConverter
from . import SeqStatusDetector
from . import SeqInformation
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
            ann_seq_processor = AnnSeqProcessor()
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
                background = ann_seq_processor.get_background(chrom)
                complete = ann_seq_processor.get_seq_with_added_type(chrom,{'other':background})
                one_hot = ann_seq_processor.get_one_hot(complete,non_conflict_type,method='order')
                one_hot.add_ann('other',ann_seq_processor.get_background(one_hot))
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
