import os
import os.path
import sys
import deepdish
import numpy as np
import pandas as pd
from time import gmtime, strftime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__+'/..'))))
from sequence_annotation.genome_handler.seq_info_parser import USCUParser
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer, AnnSeqContainer
from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
genome_information={'chromosome':{"1":22981688,"10":13272281,"11":11954808,"12":12622881,
                                  "13":13302670,"14":10246949,"15":7320470,"16":9031048,
                                  "17":12136232,"18":11077504,"19":7272499,"2":21591555,
                                  "20":3798727,"21":5834722,"3":15489435,"4":9874776,
                                  "5":13390619,"6":7024381,"7":11693588,"8":10512681,
                                  "9":10554956}
                    ,'source':'tertaodon_8_0'}
file_path = "~/deep_learning/tetraodon_8_0/raw_file/tetraodon_8_0.tsv"
principle = {'remove_end_of_strand':True,
             'with_random_choose':False,
             'replaceable':False,
             "each_region_number":{'utr_5': 300,
                                   'intergenic_region': 300,
                                   'utr_3': 300,
                                   'cds': 300,
                                   'intron': 300},
             'sample_number_per_region':1,
             'half_length':150,
             'max_diff':30
            }

frontground_types = ['cds','intron','utr_5','utr_3']
background_type = 'intergenic_region'
def get_seqinfos(chroms,id_):
    generator = SeqInfoGenerator(chroms,principle,
                                 genome_information['chromosome'],
                                 "chrom_"+id_+"_seed_"+time_root,"chrom_"+id_+"_seq_"+time_root)
    generator.generate()
    generator.seeds.to_data_frame().to_csv(root+"/"+"chrom_"+id_+"_seed.csv",
                                           index=False,sep=",", header=True)
    generator.seqs_info.to_gtf().to_csv(root+"/"+"chrom_"+id_+"_seq.gtf",
                                        index=False,sep="\t", header=False)
    return generator.seqs_info
def handle_annseq(norm_genome,seqinfos):
    ext = AnnSeqExtractor(norm_genome, seqinfos)
    ext.extract()
    np.save(root+'/answer.npy', ext.result.data_to_dict())
if __name__ == "__main__":
    time_root = strftime("%Y_%m_%d", gmtime())
    print(time_root)
    root = 'deep_learning/tetraodon_8_0/'+time_root
    if not os.path.exists(root):
        os.makedirs(root)
    df = pd.DataFrame(list(principle.items()),columns=['attribute','value'])
    df.to_csv(root+'/setting.csv',index=False)
    parser = USCUParser(file_path)
    parser.parse()
    print("AnnGenomeCreator")
    genome_file = root+'/ann_genome.h5'
    if not os.path.isfile(genome_file):
        creator = AnnGenomeCreator(genome_information,parser.result)
        creator.create()
        genome=creator.result
        deepdish.io.save(genome_file,genome.to_dict(),('zlib',9))
    else:
        genome = AnnSeqContainer()
        genome.from_dict(deepdish.io.load(genome_file))
    print("Normalize")
    norm_genome = AnnSeqContainer()
    norm_genome.ANN_TYPES = genome.ANN_TYPES
    for item in genome.data:
        norm_item = item.get_normalized(frontground_types, background_type)
        norm_genome.add(norm_item)
    print("RegionExtractor")
    total_info = SeqInfoContainer()
    for id_ in range(1,22):
        chroms_file = root+'/chrom_'+str(id_)+'.h5'
        chroms = SeqInfoContainer()
        print("prepare for extract regions:"+str(id_))
        if not os.path.isfile(chroms_file):
            for chrom in genome.data:
                if chrom.chromosome_id==str(id_):
                    extractor = RegionExtractor(chrom,frontground_types,background_type)
                    extractor.extract()
                    chroms.add(extractor.result.data)
            deepdish.io.save(chroms_file,chroms.to_dict(),('zlib',9))
        else:
            chroms.from_dict(deepdish.io.load(chroms_file))
        print("SeqInfoGenerator:"+str(id_))
        seqinfos = get_seqinfos(chroms,str(id_))
        for info in seqinfos.data:
            total_info.add(info)
    print("AnnSeqExtractor")
    handle_annseq(norm_genome,total_info)