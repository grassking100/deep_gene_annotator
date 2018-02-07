import os
import sys
import deepdish
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__+'/..'))))
from sequence_annotation.genome_handler.seq_info_parser import USCUParser
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
genome_information={'chromosome':{'chr20':3798727},'source':'tertaodon_8_0'}
file_path = "~/deep_learning/sequence_handler/raw_data/tertaodon_8.0_ensembl_genes_ch20.tsv"
principle = {'remove_end_of_strand':True,'with_random_choose':True,'replaceable':True,
             "each_region_number":{'utr_5': 8,
                                   'intergenic_region': 8,
                                   'utr_3': 8,
                                   'cds': 8,
                                   'intron': 8},
             'sample_number_per_region':5,'half_length':150,'max_diff':30
            }

chroms_info = {'chr20':3798727}
frontground_types = ['cds','intron','utr_5','utr_3']
background_type = 'intergenic_region'
if __name__ == "__main__":
    parser = USCUParser(file_path)
    parser.parse()
    creator = AnnGenomeCreator(genome_information,parser.result)
    creator.create()
    chroms = SeqInfoContainer()
    for chrom in creator.result.data:
        extractor = RegionExtractor(chrom,frontground_types,background_type)
        extractor.extract()
        chroms.add(extractor.result.data)
    generator = SeqInfoGenerator(chroms,principle,chroms_info,"seed_2018_2_6","seq_2018_2_6")
    generator.generate()
    ext = AnnSeqExtractor(creator.result, generator.seqs_info)
    ext.extract()
    generator.seeds.to_gtf().to_csv("deep_learning/seed.gtf",index=False,sep="\t", header=False)
    generator.seqs_info.to_gtf().to_csv("deep_learning/seq.gtf",index=False,sep="\t", header=False)
    deepdish.io.save('deep_learning/test.h5', ext.result.to_dict())