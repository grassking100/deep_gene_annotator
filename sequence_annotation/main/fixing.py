import os
import sys
import deepdish
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__+'/..'))))
from sequence_annotation.genome_handler.seq_info_parser import USCUParser
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
genome_information={'chromosome':{"1":22981688,"10":13272281,"11":11954808,"12":12622881,"13":13302670,"14":10246949,"15":7320470,"16":9031048,"17":12136232,"18":11077504,"19":7272499,"2":21591555,"20":3798727,"21":5834722,"3":15489435,"4":9874776,"5":13390619,"6":7024381,"7":11693588,"8":10512681,"9":10554956},'source':'tertaodon_8_0'}
file_path = "~/deep_learning/tetraodon_8_0_2.tsv"
principle = {'remove_end_of_strand':True,'with_random_choose':False,'replaceable':True,
             "each_region_number":{'utr_5': 8,
                                   'intergenic_region': 8,
                                   'utr_3': 8,
                                   'cds': 8,
                                   'intron': 8},
             'sample_number_per_region':1,'half_length':150,'max_diff':30
            }

frontground_types = ['cds','intron','utr_5','utr_3']
background_type = 'intergenic_region'
if __name__ == "__main__":
    seqs_info = pd.read_csv("deep_learning/seq.gtf",sep="\t", header=None)
    result=deepdish.io.load('deep_learning/ann_genome.h5')
    print(seqs_info)
    ext = AnnSeqExtractor(result, seqs_info)
    ext.extract()
    deepdish.io.save('deep_learning/test.h5', ext.result.to_dict(),('zlib',9))