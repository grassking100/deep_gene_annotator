import numpy as np
from . import AnnSeqTestCase
from sequence_annotation.genome_handler.exception import ProcessedStatusNotSatisfied
from sequence_annotation.genome_handler.sequence import AnnSequence,SeqInformation
from sequence_annotation.genome_handler.ann_genome_processor import get_backgrounded_genome, get_one_hot_genome,get_sub_ann_seqs
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer,SeqInfoContainer

class TestAnnGenomeProcessor(AnnSeqTestCase):
        
    def test_get_backgrounded_genome(self):
        ann_seqs = AnnSeqContainer()
        #Create sequence 1
        ann_seqs.ANN_TYPES = ['exon','intron']
        ann_seq = AnnSequence()
        ann_seq.ANN_TYPES = ['exon','intron']     
        ann_seq.length = 10
        ann_seq.id = 'input_1'
        ann_seq.strand = 'plus'
        ann_seq.chromosome_id = 'test'
        ann_seq.init_space()
        ann_seq.set_ann('exon',1,0,5).set_ann('intron',1,7,9)
        #Create sequence 2
        ann_seq2 = AnnSequence()
        ann_seq2.ANN_TYPES = ['exon','intron']     
        ann_seq2.length = 10
        ann_seq2.id = 'input_2'
        ann_seq2.strand = 'plus'
        ann_seq2.chromosome_id = 'test'
        ann_seq2.init_space()
        ann_seq2.set_ann('exon',1,0,9)
        #Add sequences
        ann_seqs.add(ann_seq)
        ann_seqs.add(ann_seq2)
        #Processing
        result = get_backgrounded_genome(ann_seqs,'other')
        except_ann_seq = AnnSeqContainer()
        except_ann_seq = AnnSequence()
        except_ann_seq.ANN_TYPES = ['exon','intron','other']     
        except_ann_seq.length = 10
        except_ann_seq.id = 'input_1'
        except_ann_seq.strand = 'plus'
        except_ann_seq.chromosome_id = 'test'
        except_ann_seq.init_space()
        except_ann_seq.set_ann('exon',1,0,5).set_ann('intron',1,7,9)
        except_ann_seq.set_ann('other',1,6,6)
        except_ann_seq2 = AnnSequence()
        except_ann_seq2.ANN_TYPES = ['exon','intron','other']     
        except_ann_seq2.length = 10
        except_ann_seq2.id = 'input_2'
        except_ann_seq2.strand = 'plus'
        except_ann_seq2.chromosome_id = 'test'
        except_ann_seq2.init_space()
        except_ann_seq2.set_ann('exon',1,0,9)
        self.assert_seq_equal(except_ann_seq,result.get('input_1'))
        self.assert_seq_equal(except_ann_seq2,result.get('input_2'))

    def test_get_one_hot_genome(self):
        ann_seqs = AnnSeqContainer()
        #Create sequence 1
        ann_seqs.ANN_TYPES = ['exon','intron','empty']
        ann_seq = AnnSequence()
        ann_seq.ANN_TYPES = ['exon','intron','empty']     
        ann_seq.length = 10
        ann_seq.id = 'input_1'
        ann_seq.strand = 'plus'
        ann_seq.chromosome_id = 'test'
        ann_seq.init_space()
        ann_seq.set_ann('exon',1,0,5).set_ann('intron',1,5,9)
        #Create sequence 2
        ann_seq2 = AnnSequence()
        ann_seq2.ANN_TYPES = ['exon','intron','empty']     
        ann_seq2.length = 10
        ann_seq2.id = 'input_2'
        ann_seq2.strand = 'plus'
        ann_seq2.chromosome_id = 'test'
        ann_seq2.init_space()
        ann_seq2.set_ann('exon',1,0,9)
        #Add sequences
        ann_seqs.add(ann_seq)
        ann_seqs.add(ann_seq2)
        #Processing
        result = get_one_hot_genome(ann_seqs,'order',['exon','intron'])
        except_ann_seq = AnnSeqContainer()
        except_ann_seq = AnnSequence()
        except_ann_seq.ANN_TYPES = ['exon','intron','empty']     
        except_ann_seq.length = 10
        except_ann_seq.id = 'input_1'
        except_ann_seq.strand = 'plus'
        except_ann_seq.chromosome_id = 'test'
        except_ann_seq.init_space()
        except_ann_seq.set_ann('exon',1,0,5).set_ann('intron',1,6,9)
        except_ann_seq.processed_status = 'one_hot'
        except_ann_seq2 = AnnSequence()
        except_ann_seq2.ANN_TYPES = ['exon','intron','empty']     
        except_ann_seq2.length = 10
        except_ann_seq2.id = 'input_2'
        except_ann_seq2.strand = 'plus'
        except_ann_seq2.chromosome_id = 'test'
        except_ann_seq2.init_space()
        except_ann_seq2.set_ann('exon',1,0,9)
        except_ann_seq2.processed_status = 'one_hot'
        self.assert_seq_equal(except_ann_seq,result.get('input_1'))
        self.assert_seq_equal(except_ann_seq2,result.get('input_2'))

    def test_extract_success(self):
        ann_seq_container = AnnSeqContainer()
        ann_seq_container.ANN_TYPES = ['exon','intron']
        """Create sequence to be extracted"""
        ann_seq = AnnSequence()
        ann_seq.ANN_TYPES = ['exon','intron']
        ann_seq.id = 'temp_plus'
        ann_seq.length = 10
        ann_seq.chromosome_id = 'temp'
        ann_seq.strand = 'plus'
        ann_seq.init_space()
        ann_seq.set_ann('exon',1,0,5).set_ann('intron',1,6,9)
        """Create real answer"""
        real_seq = AnnSequence()
        real_seq.ANN_TYPES = ['exon','intron']
        real_seq.id = 'extract'
        real_seq.length = 4
        real_seq.chromosome_id = 'temp'
        real_seq.strand = 'plus'
        real_seq.init_space()
        real_seq.set_ann('exon',1,0,0).set_ann('intron',1,1,3)
        """Create information to tell where to extract"""
        seq_info = SeqInformation()
        seq_info.chromosome_id = 'temp'
        seq_info.id = 'extract'
        seq_info.strand = 'plus'
        seq_info.start = 5
        seq_info.end = 8
        """Extract"""
        ann_seq_container.add(ann_seq)
        seq_info_container = SeqInfoContainer()
        seq_info_container.add(seq_info)
        extracted_seqs = get_sub_ann_seqs(ann_seq_container,seq_info_container)
        extracted_seq = extracted_seqs.get('extract')
        """Test"""
        self.assert_seq_equal(extracted_seq,real_seq)

