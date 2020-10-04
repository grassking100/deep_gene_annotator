import os
import sys
import deepdish as dd
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, get_subdict, write_json
from sequence_annotation.file_process.utils import read_bed, write_gff
from sequence_annotation.file_process.utils import read_fasta, BASIC_GENE_ANN_TYPES
from sequence_annotation.file_process.get_region_table import read_region_table,write_region_table,get_region_stats
from sequence_annotation.file_process.get_id_table import get_id_convert_dict
from sequence_annotation.file_process.bed2gff import bed2gff
from sequence_annotation.file_process.get_subfasta import write_fasta
from sequence_annotation.file_process.gff_analysis import main as gff_analysis_main
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.ann_genome_processor import get_mixed_genome
from sequence_annotation.genome_handler.ann_genome_processor import simplify_genome
from sequence_annotation.genome_handler.ann_genome_processor import is_one_hot_genome


def select_ann_seqs_by_length(ann_seqs, min_len=None, max_len=None, ratio=None):
    seq_lens = sorted([len(seq) for seq in ann_seqs])
    ratio = ratio or 1
    min_len = min_len or 0
    max_len = max_len or max(seq_lens)
    selected_lens = []
    for length in seq_lens:
        if min_len <= length <= max_len:
            selected_lens.append(length)
    max_len = selected_lens[:int(round(ratio * len(selected_lens)))][-1]
    selected_anns = ann_seqs.copy()
    selected_anns.clean()
    for seq in ann_seqs:
        if min_len <= len(seq) <= max_len:
            selected_anns.add(ann_seqs.get(seq.id))
    print("Total number is {}, selected number is {}, max length is {}".format(len(ann_seqs),len(selected_anns),max_len))
    return selected_anns

def classify_ann_seqs(ann_seqs):
    selected_anns = ann_seqs.copy()
    selected_anns.clean()
    multiple_exon_region_anns = selected_anns.copy()
    single_exon_region_anns = selected_anns.copy()
    no_exon_region_anns = selected_anns.copy()
    for ann_seq in ann_seqs:
        # If it is multiple exon
        if sum(ann_seq.get_ann('intron')) > 0:
            multiple_exon_region_anns.add(ann_seq)
        # If it is single exon
        elif sum(ann_seq.get_ann('exon')) > 0:
            single_exon_region_anns.add(ann_seq)
        # If there is no exon 
        else:
            no_exon_region_anns.add(ann_seq)
    data = {}
    data['multiple_exon_region'] = multiple_exon_region_anns
    data['single_exon_region'] = single_exon_region_anns
    data['no_exon_region'] = no_exon_region_anns
    return data


def select_ann_seqs_by_length_on_each_type(ann_seqs,min_len=None,max_len=None, ratio=None):
    if len(set(BASIC_GENE_ANN_TYPES) - set(ann_seqs.ANN_TYPES)) > 0:
        raise Exception("ANN_TYPES should include {}, but got {}".format(BASIC_GENE_ANN_TYPES,
                                                                         ann_seqs.ANN_TYPES))
    selected_anns = ann_seqs.copy()
    selected_anns.clean()
    classified_ann_seqs = classify_ann_seqs(ann_seqs)
    for sub_ann_seqs in classified_ann_seqs.values():
        data = select_ann_seqs_by_length(sub_ann_seqs,min_len=min_len,max_len=max_len,ratio=ratio)
        selected_anns.add(data)
    return selected_anns


def select_ann_seqs(ann_seqs, before_mix_simplify_map=None,
                   simplify_map=None, select_func=None,select_each_type=False, **kwargs):
    if select_func is None:
        if select_each_type:
            select_func = select_ann_seqs_by_length_on_each_type
        else:
            select_func = select_ann_seqs_by_length
    selected_anns = select_func(ann_seqs, **kwargs)
    if before_mix_simplify_map is not None:
        selected_anns = simplify_genome(selected_anns, before_mix_simplify_map)
    selected_anns = get_mixed_genome(selected_anns)
    if simplify_map is not None:
        selected_anns = simplify_genome(selected_anns, simplify_map)
    if not is_one_hot_genome(selected_anns):
        raise Exception("Genome is not one-hot encoded")
    return selected_anns


def get_path(output_root):
    fasta_path = os.path.join(output_root,"region.fasta")
    transcript_path = os.path.join(output_root,"transcript.gff3")
    canonical_path = os.path.join(output_root,"canonical_gene.gff3")
    stats_path = os.path.join(output_root,"region_stats.json")
    region_table_path = os.path.join(output_root,"region_table.tsv")
    return fasta_path,region_table_path,transcript_path,canonical_path,stats_path


def write_data(fasta,rna_gff,canonical_gff,region_table,output_root):
    fasta_path,region_table_path,transcript_path,canonical_path,stats_path = get_path(output_root)
    write_gff(rna_gff,transcript_path)
    write_gff(canonical_gff,canonical_path)
    write_fasta(fasta,fasta_path)
    write_region_table(region_table,region_table_path)
    write_json(get_region_stats(region_table),stats_path)
    os.system("rm {}".format(fasta_path+".fai"))
    os.system("samtools faidx {}".format(fasta_path))


def _select_ss_data(region_table,ann_seqs,fasta,rna_bed,canonical_bed,
                    id_convert_dict,canonical_id_convert_dict,output_root):
    stats_root = os.path.join(output_root,'stats')
    data_path = os.path.join(output_root,'data.h5')
    canonical_stats_root = os.path.join(stats_root,'canonical_gene_stats')
    transcript_stats_root = os.path.join(stats_root,'transcript_stats')
    create_folder(stats_root)
    chrom_ids = ann_seqs.ids
    part_rna_bed = rna_bed[rna_bed['chr'].isin(chrom_ids)]
    part_canonical_bed = canonical_bed[canonical_bed['chr'].isin(chrom_ids)]
    part_fasta = get_subdict(fasta,chrom_ids)
    part_rna_gff = bed2gff(part_rna_bed,id_convert_dict)
    part_canonical_gff = bed2gff(part_canonical_bed,canonical_id_convert_dict)
    write_data(part_fasta,part_rna_gff,part_canonical_gff,region_table,output_root)
    part_fasta_path,part_region_table_path,part_transcript_path,part_canonical_path,stats_path = get_path(output_root)
    dd.io.save(data_path,(part_fasta,ann_seqs.to_dict()))
    gff_analysis_main(part_canonical_path,part_fasta_path,canonical_stats_root,
                      chrom_source='ordinal_id_with_strand',region_table_path=part_region_table_path)
    gff_analysis_main(part_transcript_path,part_fasta_path,transcript_stats_root,
                      chrom_source='ordinal_id_with_strand',region_table_path=part_region_table_path)
    
    
def _select_ds_data(region_table,chrom_strand_ids,chrom_ids,fasta,rna_bed,canonical_bed,
                    id_convert_dict,canonical_id_convert_dict,output_root):
    part_rna_bed = rna_bed[rna_bed['chr_strand'].isin(chrom_strand_ids)]
    part_canonical_bed = canonical_bed[canonical_bed['chr_strand'].isin(chrom_strand_ids)]
    part_rna_gff = bed2gff(part_rna_bed,id_convert_dict)
    part_canonical_gff = bed2gff(part_canonical_bed,canonical_id_convert_dict)
    part_fasta = get_subdict(fasta,chrom_ids)
    write_data(part_fasta,part_rna_gff,part_canonical_gff,region_table,output_root)
        

def select_ds_data(input_result_root,split_root,id_table_path,names,output_root):
    ds_root = os.path.join(input_result_root,"double_strand")
    canonical_id_table_path = os.path.join(input_result_root,"canonical_id_convert.tsv")
    fasta = read_fasta(os.path.join(ds_root,'region.fasta'))
    rna_bed = read_bed(os.path.join(ds_root,"transcript.bed"))
    canonical_bed = read_bed(os.path.join(ds_root,"canonical_gene.bed"))
    rna_bed['chr_strand'] = rna_bed['chr'] +"_"+ rna_bed['strand']
    canonical_bed['chr_strand'] = canonical_bed['chr'] +"_"+ canonical_bed['strand']
    id_convert_dict = get_id_convert_dict(id_table_path)
    canonical_id_convert_dict = get_id_convert_dict(canonical_id_table_path)
    for name in names:
        region_table = read_region_table(os.path.join(split_root,"{}_region_table.tsv".format(name)))
        chrom_strand_ids = set(region_table['ordinal_id_wo_strand'] +"_"+ region_table['strand'])
        chrom_ids = set(region_table['ordinal_id_wo_strand'])
        name_root = os.path.join(output_root,name)
        ds_output_root = os.path.join(name_root,"double_strand")
        create_folder(name_root)
        create_folder(ds_output_root)
        _select_ds_data(region_table,chrom_strand_ids,chrom_ids,fasta,rna_bed,canonical_bed,
                        id_convert_dict,canonical_id_convert_dict,ds_output_root)
        

def select_ss_data(input_result_root,split_root,id_table_path,names,output_root,
                   min_len=None,max_len=None,ratio=None,select_each_type=False):
    ss_root = os.path.join(input_result_root,"single_strand")
    canonical_id_table_path = os.path.join(input_result_root,"canonical_id_convert.tsv")
    ann_seqs_path  = os.path.join(ss_root,"canonical_gene.h5")
    fasta = read_fasta(os.path.join(ss_root,'region.fasta'))
    rna_bed = read_bed(os.path.join(ss_root,"transcript.bed"))
    canonical_bed = read_bed(os.path.join(ss_root,"canonical_gene.bed"))
    id_convert_dict = get_id_convert_dict(id_table_path)
    canonical_id_convert_dict = get_id_convert_dict(canonical_id_table_path)
    ann_seqs = AnnSeqContainer().from_dict(dd.io.load(ann_seqs_path))
    
    for name in names:
        region_table = read_region_table(os.path.join(split_root,"{}_region_table.tsv".format(name)))
        ids = set(region_table['ordinal_id_with_strand'])
        part_ann_seqs = select_ann_seqs(ann_seqs.get_seqs(ids),min_len=min_len,max_len=max_len,
                                        ratio=ratio,select_each_type=select_each_type)
        name_root = os.path.join(output_root,name)
        ss_output_root = os.path.join(name_root,"single_strand")
        create_folder(name_root)
        create_folder(ss_output_root)
        _select_ss_data(region_table,part_ann_seqs,fasta,rna_bed,canonical_bed,
                        id_convert_dict,canonical_id_convert_dict,ss_output_root)


def load_data(path):
    data = dd.io.load(path)
    data = data[0], AnnSeqContainer().from_dict(data[1])
    return data
        
