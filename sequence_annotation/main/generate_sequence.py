import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__+'/..'))))
from sequence_annotation.utils.exception import InvalidStrandType,NegativeNumberException
from sequence_annotation.utils.exception import ReturnNoneException
from sequence_annotation.genome_handler.seq_info_parser import USCUParser
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
genome_information={'chromosome':{'chr20':3798727},'source':'tertaodon_8_0'}
file_path = "~/deep_learning/sequence_handler/raw_data/tertaodon_8.0_ensembl_genes_ch20.tsv"
principle = {'remove_end_of_strand':True,'with_random_choose':True,'replaceable':True,
             "each_region_number":{'utr_5': 80,
                                   'intergenic_region': 80,
                                   'utr_3': 80,
                                   'cds': 80,
                                   'intron': 80},
             'sample_number_per_region':5,'half_length':150,'max_diff':30
            }

##{'max_shift_length_diff': 30
## 'remove_end_of_strand': True
## 'sample_number_per_region': 5
## 'seed_id_prefix': 'chromosome_20_seed_2017_11_29'
## 'region_half_length': 150
## 'seq_id_prefix': 'chromosome_20_seq_2017_11_29'
## 'selected_target_settings': {'utr_5': 80
## 'intergenic_region': 80
## 'utr_3': 80
## 'cds': 80
## 'intron': 80}
## 'used_all_data_without_random_choose': False}

chroms_info = {'chr20':3798727}
frontground_types = ['cds','intron','utr_5','utr_3']
background_type = 'intergenic_region'
if __name__ == "__main__":
    parser = USCUParser(file_path)
    parser.parse()
    creator = AnnGenomeCreator(genome_information,parser.data)
    creator.create()
    chroms = SeqInfoContainer()
    for chrom in creator.data.data:
        extractor = RegionExtractor(chrom,frontground_types,background_type)
        extractor.extract()
        chroms.add(extractor.regions.data)
    generator = SeqInfoGenerator(chroms,principle,chroms_info,"seed_2018_2_6","seq_2018_2_6")
    generator.generate()
    generator.seeds.to_gtf().to_csv("seed.gtf",index=False,sep="\t", header=False)
    generator.seqs_info.to_gtf().to_csv("seq.gtf",index=False,sep="\t", header=False)