import json
import pandas as pd
from sequence_annotation.sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from sequence_annotation.sequence_annotation.genome_handler.ann_seq_converter import EnsemblSeqConverter
from sequence_annotation.sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.sequence_annotation.genome_handler.ann_genome_processor import AnnGenomeProcessor
from  sequence_annotation.sequence_annotation.genome_handler import ann_genome_processor
import deepdish as dd
gene_info_path = '../tetraodon_8_0_chr1_to_chr21_protein_coding.tsv'
genome_info_path = '../../genome/tetraodon_8_0/tertaodon_8_0_chrom_1_to_21.json'
with open(genome_info_path) as data_file:    
    genome_info = json.load(data_file)

converted_data = AnnSeqContainer()
converted_data.ANN_TYPES = EnsemblSeqConverter().ANN_TYPES
for chrom_id in genome_info['chromosome'].keys():
    ann_gene_container_path= "coding_sequence_preserved_external_exon_2018_07_31/chrom_"+str(chrom_id)+'_ann_gene_container_2018_07_31.h5'
    print("Load file from "+ann_gene_container_path)
    converted_data.add(dd.io.load(ann_gene_container_path))
non_conflict_type = ['cds','utr_5','utr_3','intron','other']
gene_ann_type = ['cds','utr_5','utr_3','intron']
creator = AnnGenomeCreator()
seqs=creator.create(converted_data,genome_info)
genome = AnnGenomeProcessor().get_backgrounded_genome(seqs,gene_ann_type,'other')
one_hot_genome = ann_genome_processor.AnnGenomeProcessor().get_one_hot_genome(genome,'order')
regions = AnnGenomeProcessor().get_genome_region_info(one_hot_genome)