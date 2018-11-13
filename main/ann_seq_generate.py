import deepdish as dd
import pandas as pd
import argparse
import os
import sys
sys.path.append(os.path.abspath(__file__+'/../..'))
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.ann_seq_converter import GeneticBedSeqConverter
from sequence_annotation.genome_handler.ann_seq_processor import AnnSeqProcessor
from sequence_annotation.genome_handler.sequence import AnnSequence
#Read arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-b','--bed', type=str,
                    help='Bed file path', required=True)
parser.add_argument('-o','--output', type=str,
                    help='PAth to save h5 file', required=True)
parser.add_argument('-u','--upstream', type=str,
                    help="Specific how many base pairs from upstream region to included (From 5' to 3'),default is 0",
                    default=0)
parser.add_argument('-d','--downstream', type=str,
                    help="Specific how many base pairs from downstream region to included (From 5' to 3'),default is 0",
                    default=0)
args = parser.parse_args()
bed_path = args.bed
saved_path = args.output
upstream = int(args.upstream)
downstream = int(args.downstream)

#Read bed file
bed_info = BedInfoParser().parse(pd.read_csv(bed_path,sep='\t',header=None).to_dict('record'))
#Generate annotated sequences
gene_converter = GeneticBedSeqConverter()
ann_seqs = AnnSeqContainer()
if upstream == 0 and downstream == 0:
    ann_seqs.ANN_TYPES = gene_converter.ANN_TYPES
else:
    ann_seqs.ANN_TYPES = gene_converter.ANN_TYPES + ['other']
ann_seq_processor = AnnSeqProcessor()
for seq in bed_info:
    converted_seq = gene_converter.convert(seq)
    if upstream == 0 and downstream == 0:
        ann_seqs.add(converted_seq)
    else:
        size = converted_seq.length + upstream + downstream
        expanded_seq = AnnSequence().from_dict(converted_seq.to_dict())
        expanded_seq.clean_space()
        expanded_seq.length = size
        expanded_seq.ANN_TYPES = converted_seq.ANN_TYPES + ['other']
        expanded_seq.init_space()
        for type_ in converted_seq.ANN_TYPES:
            expanded_seq.add_ann(type_,converted_seq.get_ann(type_),upstream,converted_seq.length + upstream - 1)
        background = ann_seq_processor.get_background(expanded_seq,converted_seq.ANN_TYPES)
        expanded_seq.set_ann('other',background)
        ann_seq_processor = AnnSeqProcessor()
        if not ann_seq_processor.is_one_hot(expanded_seq):
            raise Exception(str(expanded_seq.id)+" is not one-hot encoded")
        ann_seqs.add(expanded_seq)
#Save annotated sequences
dd.io.save(saved_path, ann_seqs.to_dict(),('zlib',9))

