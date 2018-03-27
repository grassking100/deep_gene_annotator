import os
import sys
import deepdish
import numpy as np
import pandas as pd
from time import gmtime, strftime
sys.path.append(os.path.abspath(__file__+'/../..'))
from sequence_annotation.genome_handler.seq_info_parser import USCUParser
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer, AnnSeqContainer
from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
file_root="whole_genome_"
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
             'replaceable':True,
             "each_region_number":{'utr_5': 326842,
                                   'intergenic_region': 30429,
                                   'utr_3': 248745,
                                   'cds': 4566,
                                   'intron': 4177},
             'sample_number_per_region':1,
             'half_length':150,
             'max_diff':30
            }

frontground_types = ['cds','intron','utr_5','utr_3']
background_type = 'intergenic_region'
def get_seqinfos(chroms):
    generator = SeqInfoGenerator(chroms,principle,
                                 genome_information['chromosome'],
                                 file_root+"seed_"+time_root,file_root+"seq_"+time_root)
    generator.generate()
    generator.seeds.to_data_frame().to_csv(root+"/"+file_root+"seed.csv",
                                           index=False,sep=",", header=True)
    generator.seqs_info.to_gtf().to_csv(root+"/"+file_root+"seq.gtf",
                                        index=False,sep="\t", header=False)
    return generator.seqs_info
def handle_annseq(norm_genome,seqinfos):
    ext = AnnSeqExtractor(norm_genome, seqinfos)
    ext.extract()
    print("Save file to "+root+'/'+file_root+'answer.npy')
    np.save(root+'/'+file_root+'answer.npy', ext.result.data_to_dict())
def prepare_genome(parser_result):
    print("Genome")
    genome_file = root+'/ann_genome.h5'
    if not os.path.isfile(genome_file):
        creator = AnnGenomeCreator(genome_information,parser_result)
        creator.create()
        genome=creator.result
        print("Save file to "+genome_file)
        deepdish.io.save(genome_file,genome.to_dict(),('zlib',9))
    else:
        print("Load file from "+genome_file)
        genome = AnnSeqContainer()
        genome.from_dict(deepdish.io.load(genome_file))
    return genome
def prepare_norm_genome(parser_result):
    print("Normalize")
    norm_genome_file = root+'/norm_genome.h5'
    if not os.path.isfile(norm_genome_file):
        genome = prepare_genome(parser_result)
        norm_genome = AnnSeqContainer()
        norm_genome.ANN_TYPES = genome.ANN_TYPES
        for item in genome.data:
            norm_item = item.get_normalized(frontground_types, background_type)
            norm_genome.add(norm_item)
        print("Save file to "+norm_genome_file)
        deepdish.io.save(norm_genome_file,norm_genome.to_dict(),('zlib',9))
    else:
        print("Load file from "+norm_genome_file)
        norm_genome = AnnSeqContainer()
        norm_genome.from_dict(deepdish.io.load(norm_genome_file))
    return norm_genome
def prepare_total_info(parser_result):
    print("total_info")
    genome = None
    chroms = SeqInfoContainer()
    for id_ in range(1,22):
        chroms_file = root+'/chrom_'+str(id_)+'.h5'
        print("prepare for extract regions:"+str(id_))
        if not os.path.isfile(chroms_file):
            if genome is None:
                genome = prepare_genome(parser_result)
            for chrom in genome.data:
                if chrom.chromosome_id==str(id_):
                    extractor = RegionExtractor(chrom,frontground_types,background_type)
                    extractor.extract()
                    chroms.add(extractor.result)
            print("Save file to "+chroms_file)
            deepdish.io.save(chroms_file,chroms.to_dict(),('zlib',9))
        else:
            print("Load file from "+chroms_file)
            temp = SeqInfoContainer()
            temp.from_dict(deepdish.io.load(chroms_file))
            chroms.add(temp)
    print("SeqInfoGenerator")
    seqinfos = get_seqinfos(chroms)
    return seqinfos
def prepare_annseq(parser_result):
    print("AnnSeqExtractor")
    norm_genome=prepare_norm_genome(parser_result)
    total_info=prepare_total_info(parser_result)
    handle_annseq(norm_genome,total_info)
if __name__ == "__main__":
    time_root = strftime("%Y_%m_%d", gmtime())
    print(time_root)
    root = os.path.expanduser('~/deep_learning/tetraodon_8_0/' + time_root)
    if not os.path.exists(root):
        os.makedirs(root)
    df = pd.DataFrame(list(principle.items()),columns=['attribute','value'])
    df.to_csv(root+'/setting.csv',index=False)
    parser = USCUParser(file_path)
    parser.parse()
    prepare_annseq(parser.result)
    print("End of program")
