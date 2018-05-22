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
def get_wanted_seq(chrom):
    ann_seq_processor = AnnSeqProcessor()
    background = ann_seq_processor.get_background(chrom)
    complete = ann_seq_processor.combine_status(chrom,{'other':background})
    one_hot = ann_seq_processor.get_one_hot(complete,non_conflict_type,method='order')
    further_seq = exon_handler.further_division(one_hot)
    internal_seq = exon_handler.discard_external(further_seq)
    simple_seq = exon_handler.simplify_exon_name(internal_seq)
    simple_seq.add_ann('other',ann_seq_processor.get_background(simple_seq))
    if not ann_seq_processor.is_full_annotated(simple_seq,non_conflict_type):
        raise Exception("Chromomsome is not fully annotated")
    return simple_seq
if __name__=="__main__":
    data = {'21':5834722,"20":3798727,"15":7320470,"19":7272499,"6":7024381}
    for chrom_id,chrom_length in data.items():
        #chrom_id = str(21)#input('Chromosome id:')
        #chrom_length = 5834722#int(input('Chromosome length:'))
        chrom_source = "ensembl"#input('Chromosome source:')
        id_prefix = "tetraodon_8_0_chr"+chrom_id+"_protein_coding_"#input('Id prefix:')
        length = 10000#int(input('Length:'))
        ensembl_path = abspath(expanduser("merged_tetraodon_8_0_chr"+chrom_id+"_protein_coding.tsv"))#input('Ensembl file path:')))
        saved_path = "ensembl_tetraodon_8_0_chr"+chrom_id+"_protein_coding_range_"+str(length)#input('Saved file name:')))
        seq_info = {'chromosome':{chrom_id:chrom_length},'source':chrom_source}
        print(saved_path)
        print(ensembl_path)
        id_ = 0
        #Create object to use
        ann_seq_container = AnnSeqContainer()
        info_container = SeqInfoContainer()
        ann_seq_container.ANN_TYPES = non_conflict_type + ['exon']
        ann_seq_processor = AnnSeqProcessor()
        gene_converter = EnsemblSeqConverter(extra_types=['exon'])
        ann_genome_creator = AnnGenomeCreator()
        converted_data = AnnSeqContainer()
        exon_handler = ExonHandler()
        #Read data
        data = pd.read_csv(ensembl_path,sep='\t').to_dict('record')
        #Create Annotated Genoome
        converted_data.ANN_TYPES = gene_converter.ANN_TYPES
        for seq in EnsemblInfoParser().parse(data):
            converted_seq = gene_converter.convert(seq)
            converted_data.add(converted_seq)
        genome=ann_genome_creator.create(converted_data,seq_info)
        #Get one strand
        for chrom in genome.data:
            #Create sequence with wanted annotation
            wanted_seq = get_wanted_seq(chrom)
            ann_seq_container.add(wanted_seq)
            #Create regions for extraction
            for index in range(math.ceil(chrom.length/length)):
                info = SeqInformation()
                info.id = id_prefix + str(id_)
                id_ += 1
                info.chromosome_id = chrom.chromosome_id
                info.strand = chrom.strand
                info.start = index*length
                info.end = (index+1)*length
                info.source = chrom_source
                info.note = '2018_5_21'
                if info.end >= chrom.length:
                    info.end = chrom.length - 1
                    info.start = info.end - length + 1
                info_container.add(info)
        #Extract sequence    
        extracted = AnnSeqExtractor().extract(ann_seq_container,info_container)
        #Convert to storing format
        answer = {}
        for item in extracted.to_dict()['data']:
            data = {}
            for type_ in non_conflict_type:
                data[type_] = item['data'][type_]
            answer[item['id']]=data
        gtf_path = abspath(expanduser("raw_data_"+saved_path+".gtf"))
        tsv_path = abspath(expanduser("raw_data_"+saved_path+".tsv"))
        npy_path = abspath(expanduser(saved_path+".npy"))
        info_container.to_data_frame().to_csv(tsv_path,index=None,sep='\t')
        info_container.to_gtf().to_csv(gtf_path,index=None,sep='\t',header=None)
        deepdish.io.save(npy_path, answer,('zlib',9))
        command = ("bedtools getfasta -s -name -fi Tetraodon_nigroviridis.TETRAODON8.dna_sm."
                   "chromosome."+chrom_id+".fa -bed "+gtf_path + " -fo " 
                   "TETRAODON8_sm_chrom_"+chrom_id+"_selected_region.fasta")
        print(command)
        #os.system(command)
