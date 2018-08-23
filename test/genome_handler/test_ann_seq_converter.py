import unittest
from . import AnnSeqTestCase
from sequence_annotation.genome_handler.ann_seq_converter import EnsemblSeqConverter
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.exception import NotOneHotException
class TestAnnSeqConverter(AnnSeqTestCase):
    def test_ensembl_data(self):
        converter = EnsemblSeqConverter()
        data = {'strand':'plus','protein_id':'test','chrom':1,
                'tx_start':10,'tx_end':300,
                'exons_start' :[10,50,100,150,200],
                'exons_end':[40,90,140,190,300],
                'utrs_5_start':[10,50],
                'utrs_5_end':[40,60],
                'cdss_start'  :[61,100,150],
                'cdss_end':[90,140,160],
                'utrs_3_start':[161,200],
                'utrs_3_end':[190,300],
               }
        ann_seq = AnnSequence()
        ann_seq.length = 291
        ann_seq.id = 'test'
        ann_seq.chromosome_id = 1
        ann_seq.strand = 'plus'
        ann_seq.ANN_TYPES = ['cds','intron','utr_5','utr_3']
        ann_seq.init_space()
        ann_seq.set_ann('utr_5',1,0,30).set_ann('utr_5',1,40,50)
        ann_seq.set_ann('cds',1,51,80).set_ann('cds',1,90,130).set_ann('cds',1,140,150)
        ann_seq.set_ann('utr_3',1,151,180).set_ann('utr_3',1,190,290)
        ann_seq.set_ann('intron',1,31,39).set_ann('intron',1,81,89)
        ann_seq.set_ann('intron',1,131,139).set_ann('intron',1,181,189)
        result = converter.convert(data)
        self.assert_seq_equal(ann_seq,result)
    def test_ensembl_data_not_complete(self):
        converter = EnsemblSeqConverter()
        data = {'strand':'plus','protein_id':'test','chrom':1,
                'tx_start':10,'tx_end':300,
                'exons_start' :[10,50,100,150,200],
                'exons_end':[40,90,140,190,300],
                'utrs_5_start':[10,50],
                'utrs_5_end':[40,60],
                'cdss_start'  :[61,100,150,160],
                'cdss_end':[90,140,160],
                'utrs_3_start':[161,200],
                'utrs_3_end':[190,300],
               }
        with self.assertRaises(Exception):
            result = converter.convert(data)
    def test_ensembl_data_not_filled_correctly(self):
        converter = EnsemblSeqConverter()
        data = {'strand':'plus','protein_id':'test','chrom':1,
                'tx_start':10,'tx_end':300,
                'exons_start' :[10,50,100,150,200],
                'exons_end':[40,90,140,190,300],
                'utrs_5_start':[10,50],
                'utrs_5_end':[40,60],
                'cdss_start'  :[61,100,150],
                'cdss_end':[90,140,160],
                'utrs_3_start':[161,200],
                'utrs_3_end':[191,300],
               }
        with self.assertRaises(Exception):
            result = converter.convert(data)
    def test_ensembl_data_is_not_one_hot(self):
        converter = EnsemblSeqConverter()
        data = {'strand':'plus','protein_id':'test','chrom':1,
                'tx_start':10,'tx_end':300,
                'exons_start' :[10,50,100,150,200],
                'exons_end':[40,90,140,190,300],
                'utrs_5_start':[10,10,50],
                'utrs_5_end':[40,40,60],
                'cdss_start'  :[61,100,150],
                'cdss_end':[90,140,160],
                'utrs_3_start':[161,200],
                'utrs_3_end':[190,300],
               }
        with self.assertRaises(Exception):
            result = converter.convert(data)
