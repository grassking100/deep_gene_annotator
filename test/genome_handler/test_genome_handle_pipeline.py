import math
import unittest
import os
from sequence_annotation.file_process.seq_info_parser import BedInfoParser
from sequence_annotation.genome_handler.exon_handler import ExonHandler
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer, SeqInfoContainer
from sequence_annotation.genome_handler.ann_seq_processor import get_background, get_seq_with_added_type, get_one_hot, simplify_seq
from sequence_annotation.genome_handler.ann_seq_converter import CodingBedSeqConverter
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.genome_handler.ann_genome_processor import get_sub_ann_seqs
from sequence_annotation.genome_handler.sequence import SeqInformation

bed_root = os.path.join(os.path.dirname(__file__), '..','data','bed')

class TestGenomeHandlePipeline(unittest.TestCase):
    def test_runable(self):
        try:
            id_ = 0
            length = 180
            # Create object to use
            info_container = SeqInfoContainer()
            gene_converter = CodingBedSeqConverter()
            ann_genome_creator = AnnGenomeCreator()
            converted_data = AnnSeqContainer()
            exon_handler = ExonHandler()
            ensembl_path = os.path.join(bed_root, 'simple.bed')
            seq_info = {'chromosome': {'chr1': 240}, 'source': 'test'}
            # Create Annotated Genoome
            map_ = {'exon': ['cds', 'utr_5', 'utr_3'], 'intron': ['intron']}
            for seq in BedInfoParser().parse(ensembl_path):
                converted_seq = gene_converter.convert(seq)
                converted_seq = simplify_seq(converted_seq, map_)
                further_seq = exon_handler.further_division(converted_seq)
                internal_seq = exon_handler.discard_external(further_seq)
                simple_seq = exon_handler.simplify_exon_name(internal_seq)
                if converted_data.ANN_TYPES is None:
                    converted_data.ANN_TYPES = simple_seq.ANN_TYPES
                converted_data.add(simple_seq)

            genome = ann_genome_creator.create(converted_data, seq_info)
            ann_seq_container = AnnSeqContainer(['intron', 'other', 'exon'])
            # Get one strand
            for chrom in genome.data:
                # Create sequence with wanted annotation
                background = get_background(chrom)
                complete = get_seq_with_added_type(chrom,
                                                   {'other': background})
                one_hot = get_one_hot(complete, ['intron', 'other', 'exon'],
                                      method='order')
                one_hot.add_ann('other', get_background(one_hot))
                ann_seq_container.add(one_hot)
                # Create regions for extraction
                for index in range(math.ceil(simple_seq.length / length)):
                    info = SeqInformation()
                    info.id = id_
                    id_ += 1
                    info.chromosome_id = chrom.chromosome_id
                    info.strand = chrom.strand
                    info.start = index * length
                    info.end = (index + 1) * length - 1
                    if info.end >= simple_seq.length:
                        info.end = simple_seq.length - 1
                        info.start = info.end - length + 1
                    info_container.add(info)
            # Extract sequence
            extracted = get_sub_ann_seqs(ann_seq_container, info_container)
            # Convert to storing format
            answer = {}
            for item in extracted.to_dict()['data']:
                data = {}
                for type_ in ['intron', 'other', 'exon']:
                    data[type_] = item['data'][type_]
                answer[item['id']] = data
        except Exception:
            self.fail("There are some unexpected exception occur.")
