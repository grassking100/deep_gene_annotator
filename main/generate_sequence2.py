import os
import sys
sys.path.append(os.path.abspath(__file__+'/../..'))
import json
import math
import unittest
import deepdish
import numpy as np
import pandas as pd
import math
from time import gmtime, strftime
from os.path import abspath, expanduser
from sequence_annotation.genome_handler.seq_info_parser import UscuInfoParser, EnsemblInfoParser
from sequence_annotation.genome_handler.ann_genome_creator import AnnChromCreator, AnnGenomeCreator
from sequence_annotation.genome_handler.exon_handler import ExonHandler
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer, AnnSeqContainer
from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
from sequence_annotation.genome_handler.ann_seq_processor import AnnSeqProcessor
from sequence_annotation.genome_handler.ann_seq_converter import UscuSeqConverter, EnsemblSeqConverter
from sequence_annotation.data_handler.fasta_converter import FastaConverter
from sequence_annotation.data_handler.seq_converter import SeqConverter
from sequence_annotation.genome_handler.seq_status_detector import SeqStatusDetector
from sequence_annotation.genome_handler.sequence import SeqInformation
non_conflict_type = ['cds','utr_5','utr_3','intron','other']
gene_ann_type = ['cds','utr_5','utr_3','intron']
length = 10000#int(input('Length:'))
half_length = int(length/2)
def get_wanted_seq(chrom):
    ann_seq_processor = AnnSeqProcessor()
    background = ann_seq_processor.get_background(chrom,gene_ann_type)
    complete = ann_seq_processor.combine_status(chrom,{'other':background})
    one_hot = ann_seq_processor.get_one_hot(complete,non_conflict_type,method='order')
    if not ann_seq_processor.is_one_hot(one_hot,non_conflict_type):
        raise Exception("Chromomsome is not one hot")
    return one_hot
def get_region(genome):
    info_container = SeqInfoContainer()
    for chrom in genome:
    #Create sequence with wanted annotation
        id_ = 0
        wanted_seq = get_wanted_seq(chrom)
        ann_seq_container.add(wanted_seq)
        #Create regions for extraction
        for index in range(math.ceil(chrom.length/half_length)):
            info = SeqInformation()
            info.id = id_prefix + chrom.strand + "_" +str(id_)
            id_ += 1
            info.chromosome_id = chrom.chromosome_id
            info.strand = chrom.strand
            info.start = index*half_length
            info.end =   info.start + length - 1
            info.source = chrom_source
            info.note = '2018_5_30'
            if info.end >= chrom.length:
                info.end = chrom.length - 1
                info.start = info.end - length + 1
            info_container.add(info)
    return info_container
if __name__=="__main__":
    chrom_info = {'21':5834722,"20":3798727,"15":7320470,"19":7272499,"6":7024381}        
    #chrom_info = {"20":3798727}
    ensembl_path = abspath(expanduser("merged_tetraodon_8_0_chr{chrom_id}_protein_coding.tsv"))#input('Ensembl file path:')))
    saved_path = "ensembl_tetraodon_8_0_chr{chrom_id}_protein_coding_range_{length}_{status_string}"#input('Saved file name:')))
    region_path = "ensembl_tetraodon_8_0_chr{chrom_id}_protein_coding_region_{status_string}"#input('Saved file name:')))
    for discard_external_status in [True,False]:
        if discard_external_status:
            status_string = "discard_external_exon"
        else:
            status_string = "preserved_external_exon"
        for chrom_id,chrom_length in chrom_info.items():
            #chrom_id = str(21)#input('Chromosome id:')
            #chrom_length = 5834722#int(input('Chromosome length:'))
            chrom_source = "ensembl"#input('Chromosome source:')
            id_prefix = "tetraodon_8_0_chr"+chrom_id+"_protein_coding_"#input('Id prefix:')
            seq_info = {'chromosome':{chrom_id:chrom_length},'source':chrom_source}
            #Create object to use
            ann_seq_container = AnnSeqContainer()
            ann_seq_container.ANN_TYPES = non_conflict_type + ['exon']
            ann_seq_processor = AnnSeqProcessor()
            gene_converter = EnsemblSeqConverter(extra_types=['exon'])
            ann_genome_creator = AnnGenomeCreator()
            converted_data = AnnSeqContainer()
            #Read data
            data = pd.read_csv(ensembl_path.format(chrom_id=chrom_id),sep='\t').to_dict('record')
            exon_handler = ExonHandler()
            #Create Annotated Genoome
            converted_data.ANN_TYPES = gene_converter.ANN_TYPES
            for seq in EnsemblInfoParser().parse(data):
                converted_seq = gene_converter.convert(seq)
                if discard_external_status:
                    if not ann_seq_processor.is_one_hot(converted_seq,gene_ann_type):
                        raise Exception("Chromomsome is not one hot")
                    else:
                        converted_seq.processed_status = "one_hot"
                    further_seq = exon_handler.further_division(converted_seq)
                    internal_seq = exon_handler.discard_external(further_seq)
                    simple_seq = exon_handler.simplify_exon_name(internal_seq)
                    converted_data.add(simple_seq)
                else:
                    converted_data.add(converted_seq)
            genome=ann_genome_creator.create(converted_data,seq_info)
            #Get one strand
            region_container = SeqInfoContainer()
            temp = []
            for chrom in genome:
                temp += [RegionExtractor().extract(get_wanted_seq(chrom))]
            for info_list in temp:
                for info in info_list:
                    region_container.add(info)
            temp = region_path.format(chrom_id=chrom_id,status_string=status_string)
            region_container.to_data_frame().to_csv(temp+".tsv",index=None,sep='\t')
            info_container = get_region(genome)
            #Extract sequence    
            extracted = AnnSeqExtractor().extract(ann_seq_container,info_container)
            #Convert to storing format
            answer = {}
            for item in extracted:
                if not ann_seq_processor.is_one_hot(item,non_conflict_type):
                    raise Exception(str(item.id)+" is not one hot")
            for item in extracted.to_dict()['data']:
                data = {}
                for type_ in non_conflict_type:
                    data[type_] = item['data'][type_]
                answer[item['id']]=data
            temp = saved_path.format(chrom_id=chrom_id,length=length,status_string=status_string)
            gtf_path = abspath(expanduser("gtf/"+temp+".gtf"))
            tsv_path = abspath(expanduser("tsv/"+temp+".tsv"))
            h5_path = abspath(expanduser("h5/"+temp+".h5"))
            info_container.to_data_frame().to_csv(tsv_path,index=None,sep='\t')
            info_container.to_gtf().to_csv(gtf_path,index=None,sep='\t',header=None)
            deepdish.io.save(h5_path, answer,('zlib',9))
            command = ("bedtools getfasta -s -name -fi Tetraodon_nigroviridis.TETRAODON8.dna_sm."
                       "chromosome."+chrom_id+".fa -bed "+gtf_path + " -fo " 
                       "TETRAODON8_sm_chrom_"+chrom_id+"_"+status_string+"_region.fasta")
            os.system(command)
    for discard_external_status in [True,False]:
        all_data = {}
        if discard_external_status:
            status_string = "discard_external_exon"
        else:
            status_string = "preserved_external_exon"
        for chrom_id,chrom_length in chrom_info.items():
            temp = saved_path.format(chrom_id=chrom_id,length=length,status_string=status_string)
            h5_path = abspath(expanduser("h5/"+temp+".h5"))
            temp = deepdish.io.load(h5_path)
            all_data.update(temp)
        id_list = "_".join(list(chrom_info.keys()))
        all_data_path = 'ensembl_tetraodon_8_0_chr{id_list}_protein_coding_range_{length}_{status_string}'
        np.save(all_data_path.format(id_list=id_list,
                                     length=length,
                                     status_string=status_string)+'.npy',
                all_data)