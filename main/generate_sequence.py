import os
import sys
import deepdish
import numpy as np
from time import gmtime, strftime
sys.path.append(os.path.abspath(__file__+'/../..'))
from sequence_annotation.genome_handler.seq_info_parser import UscuInfoParser,EnsemblInfoParser
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer, AnnSeqContainer
from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
from sequence_annotation.genome_handler.ann_seq_converter import UscuSeqConverter,EnsemblSeqConverter
import json
import pandas as pd
frontground_types = ['cds','intron','utr_5','utr_3']
background_type = 'other'
def get_seqinfos(chroms):
    print("Prepare seqinfos")
    generator = SeqInfoGenerator()
    generator.generate(chroms,principle,genome_info['chromosome'],"seed_"+time_root,"seq_"+time_root)
    generator.seeds.to_data_frame().to_csv(root+"/"+"seed.csv",
                                           index=False,sep=",", header=True)
    generator.seqs_info.to_gtf().to_csv(root+"/"+"seq.gtf",
                                        index=False,sep="\t", header=False)
    return generator.seqs_info
def handle_annseq(norm_chrom,seq_infos,file_name):
    ext = AnnSeqExtractor(norm_chrom, seq_infos)
    ext.extract()
    print("Save file to "+root+'/'+file_name)
    deepdish.io.save(root+'/'+file_name, ext.result.data_to_dict(),('zlib',9))
    print("Save file done")
def prepare_chromosome(parser_result,chrom_id):
    print("Chromosome:"+str(chrom_id))
    chrom_file = root+'/ann_'+chrom_id+'.npy'
    region_data = []
    for data in parser_result:
        if data['chrom']==chrom_id:
            region_data.append(data)
    chrom_info = dict(genome_info)
    chrom_info['chromosome'] = {chrom_id:genome_info['chromosome'][chrom_id]}
    if not os.path.isfile(chrom_file):
        creator = AnnGenomeCreator(chrom_info,region_data)
        chromosome=creator.create()
        print("Save file to "+chrom_file)
        deepdish.io.save(chrom_file, chromosome.to_dict(),('zlib',9))
        print("Save file done")
    else:
        print("Load file from "+chrom_file)
        chromosome = AnnSeqContainer()
        chromosome.from_dict(deepdish.io.load(chrom_file))
        print("Load done")
    return chromosome
def prepare_norm_chromosome(parser_result,chrom_id,is_return):
    print("Normalized chromosome:"+str(chrom_id))
    norm_chrom_file = root+'/norm_'+chrom_id+'.npy'
    if not os.path.isfile(norm_chrom_file):
        chrom = prepare_chromosome(parser_result,chrom_id)
        norm_chrom = AnnSeqContainer()
        norm_chrom.ANN_TYPES = chrom.ANN_TYPES
        for item in chrom.data:
            norm_item = item.get_normalized(frontground_types, background_type)
            norm_chrom.add(norm_item)
        print("Save file to "+norm_chrom_file)
        deepdish.io.save(norm_chrom_file, norm_chrom.to_dict(),('zlib',9))
        print("Save file done")
    elif is_return:
        print("Load file from "+norm_chrom_file)
        norm_chrom = AnnSeqContainer()
        norm_chrom.from_dict(deepdish.io.load(norm_chrom_file))
        print("Load done")
    else:
        norm_chrom = None
    return norm_chrom
def prepare_total_info(parser_result):
    print("total_info")
    genome = SeqInfoContainer()
    for id_ in genome_info['chromosome'].keys():
        chroms_file = root+'/'+str(id_)+'_region_info.npy'
        print("prepare for extract regions:"+str(id_))
        if not os.path.isfile(chroms_file):
            chrom_info = SeqInfoContainer()
            chrom = prepare_norm_chromosome(parser_result,id_,True)
            for strand in chrom.data:
                extractor = RegionExtractor(strand,frontground_types,background_type)
                extractor.extract()
                chrom_info.add(extractor.result)
            print("Save file to "+chroms_file)
            deepdish.io.save(chroms_file, chrom_info.to_dict(),('zlib',9))
            print("Save file done")
        else:
            print("Load file from "+chroms_file)
            chrom_info = SeqInfoContainer()
            chrom_info.from_dict(deepdish.io.load(chroms_file))
            print("Load done")
        genome.add(chrom_info)
    print("SeqInfoGenerator")
    seqinfos = get_seqinfos(genome)
    return seqinfos
def prepare_annseq(parser_result):
    print("AnnSeqExtractor")
    total_info=prepare_total_info(parser_result)
    for id_ in genome_info['chromosome'].keys():
        chrom = prepare_norm_chromosome(parser_result,id_,is_return=True)
        chrom_info = dict(total_info)
        chrom_info['chromosome']= {id_:genome_info['chromosome'][id_]}
        handle_annseq(chrom,chrom_info,'ann_seq_of_'+id_+'.npy')
def get_bool(string):
    string = string.lower()
    if string == "t" or string == "true":
        return True
    elif string == "f" or string == "false":
        return False
    else:
        raise Exception("Invalid input:"+string)
if __name__ == "__main__":
    time_root = strftime("%Y_%m_%d", gmtime())
    print(time_root)
    path_root = os.path.expanduser(input("Please input folder to saved the data:"))
    folder = input("Please input folder name prefix:")
    root = path_root+'/' + folder + "_" + time_root
    if not os.path.exists(root):
        print("Create folder:"+root)
        os.makedirs(root)
    print("Input setting...")
    principle = {}
    principle['remove_end_of_strand']=get_bool(input('remove end of strand (T/F):'))
    principle['with_random_choose']=not get_bool(input('use all region (T/F):'))
    if principle['with_random_choose']:
        principle['replaceable'] = get_bool(input('replaceable (T/F):'))
        principle['each_region_number'] = {}
        print("Input each region number:")
        regions = {'utr_5':'5\' UTR','utr_3':'3\' UTR','cds':'CDS','intron':'intron','other':'other'}
        for key,value in regions.items():
            principle['each_region_number'][key] = int(input('\t'+value+':'))
    principle['sample_number_per_region'] = int(input('sample number per annotation region:'))
    principle['constant_length'] = get_bool(input('constant length(T/F):'))
    if principle['length_constant']: 
        principle['total_length'] = int(input('total length:'))
    else:
        principle['max_diff'] = int(input('max expand length per end:'))
    principle['half_length'] = int(input('core\'s half length:'))
    data_source = input('Data source:')
    file_path = os.path.expanduser(input("Table path:"))
    genome_info_path = os.path.expanduser(input('Genome infomration data path:'))
    with open(genome_info_path) as data_file:    
        genome_info = json.load(data_file)
    setting = {'principle':principle,
               'genome_info_path':genome_info_path,
               'UCSC_file_path':file_path,
               'data_source':data_source}
    with open(root+'/setting.json', 'w') as handle:
        json.dump(setting,handle,sort_keys=True ,indent=4)
    data = pd.read_csv(file_path,sep='\t').to_dict('record')
    if data_source=='uscu':
        parser = UscuInfoParser()
        gene_converter = UscuSeqConverter()
    elif data_source=='ensembl':
        parser = UscuInfoParser()
        gene_converter = EnsemblSeqConverter()
        parser = EnsemblInfoParser()
    converted_data = []
    for seq in parser.parse(data):
        converted_seq = gene_converter.convert(seq)
        converted_data.append(converted_seq)
    prepare_annseq(parser.parse(converted_data))
    print("End of program")
