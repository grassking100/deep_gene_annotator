import os
import sys
import pandas as pd
from argparse import ArgumentParser
from pandas.errors import EmptyDataError
ROOT = os.path.dirname(__file__)+"/../.."
sys.path.append(ROOT)
from sequence_annotation.utils.utils import write_json, create_folder
from sequence_annotation.file_process.utils import BED_COLUMNS, read_bed, write_bed, write_gff
from sequence_annotation.file_process.get_id_table import get_id_convert_dict
from sequence_annotation.file_process.bed2gff import bed2gff
from sequence_annotation.file_process.gff2bed import gff2bed
from sequence_annotation.file_process.get_region_table import read_region_table
from sequence_annotation.file_process.get_region_table import main as region_id_conversion_main
from sequence_annotation.file_process.rename_chrom import main as rename_chrom_main
from sequence_annotation.preprocess.ann_seqs_summary import main as ann_seqs_summary_main
from sequence_annotation.preprocess.create_ann_genome import main as create_ann_genome_main
from sequence_annotation.preprocess.recurrent_cleaner import recurrent_cleaner
from sequence_annotation.preprocess.get_cleaned_code_bed import main as get_cleaned_code_bed_main
from sequence_annotation.preprocess.region_gene_count_filter import main as region_gene_count_filter_main
from sequence_annotation.preprocess.redefine_coordinate import redefine_coordinate
from sequence_annotation.preprocess.create_canonical_gene import main as create_canonical_gene_main


BASH_ROOT = ROOT+'/bash'
_SLOP_COMMAND = "bedtools slop -s -i {} -g {} -l {} -r {} > {}"
_GENE_TO_REGION_MAPPING_COMMAND = "bash "+BASH_ROOT+"/gene_to_region_mapping.sh -i {} -d {} -o {} -c {}"
_SORT_MERGE_COMMAND = "bash "+BASH_ROOT+"/sort_merge.sh -i {} -o {}"


def _convert_bed_id(bed, id_convert_dict):
    query = 'id'
    returned = bed.copy()
    returned[query] = returned[query].replace(id_convert_dict)
    returned = returned.sort_values(['id'])
    return returned


def _get_fasta(input_path,bed_path,output_path,use_name=False,use_strand=False):
    command = "bedtools getfasta "
    if use_name:
        command += '-name '
    if use_strand:
        command += '-s '
    command += "-fi {} -bed {} -fo {}"
    os.system(command.format(input_path,bed_path,output_path))
    
    
def _get_coord_id(bed):
    ids = bed['chr'] + "_" + bed['strand'] + "_" + bed['start'].astype(str) + "_" + bed['end'].astype(str)
    return ids


def _get_fai(fasta_path):
    os.system("rm {}".format(fasta_path+".fai"))
    os.system("samtools faidx {}".format(fasta_path))

    
def _get_bed_num(path):
    try:
        return len(read_bed(path,ignore_strand_check=True))
    except EmptyDataError:
        return 0
    
def main(input_bed_path,background_bed_path,id_table_path,genome_path,
         upstream_dist,downstream_dist,source_name,
         output_root,merge_overlapped=False):
    #Set path
    cleaning_root=os.path.join(output_root,"cleaning")
    region_root=os.path.join(output_root,"region")
    result_root=os.path.join(output_root,"result")
    ss_root=os.path.join(result_root,"single_strand")
    ds_root=os.path.join(result_root,"double_strand")
    create_folder(output_root)
    create_folder(cleaning_root)
    create_folder(region_root)
    create_folder(result_root)
    create_folder(ss_root)
    create_folder(ds_root)
    #
    stats_path = os.path.join(output_root,"count.json")
    #
    recurrent_cleaned_gene_path = os.path.join(cleaning_root,"recurrent_cleaned_gene.bed")
    recurrent_cleaned_transcript_path = os.path.join(cleaning_root,"recurrent_cleaned_transcript.bed")
    #
    potential_region_path=os.path.join(region_root,"potential_region.bed")
    potential_region_fasta_path=os.path.join(region_root,"potential_region.fasta")
    mapping_result_path = os.path.join(region_root,"gene_region_mapping.bed")
    mapping_count_path = os.path.join(region_root,"gene_region_mapping_count.tsv")
    cleaned_region_path=os.path.join(region_root,"cleaned_region.bed")
    dirty_region_path=os.path.join(region_root,"dirty_region.bed")    
    final_origin_region_bed_path=os.path.join(region_root,"final_origin_region.bed")
    multi_gene_region_path=os.path.join(region_root,"multi_gene_region.bed")
    #
    final_origin_strand_region_bed_path=os.path.join(result_root,"final_origin_strand_region.bed")
    region_table_path=os.path.join(result_root,"region_table.tsv")
    #
    region_bed_path=os.path.join(ss_root,"ss_region.bed")
    region_fasta_path=os.path.join(ss_root,"ss_region.fasta")
    rna_bed_path = os.path.join(ss_root,"ss_rna.bed")
    rna_gff_path = os.path.join(ss_root,"ss_rna.gff3")
    canonical_bed_path=os.path.join(ss_root,"ss_canonical.bed")
    canonical_gff_path=os.path.join(ss_root,"ss_canonical.gff3")
    canonical_h5_path=os.path.join(ss_root,"ss_canonical.h5")
    canonical_length_stats_root=os.path.join(ss_root,"ss_canonical_length_stats")
    alt_region_gff_path=os.path.join(ss_root,"ss_alt_region.gff3")
    alt_region_h5_path=os.path.join(ss_root,"ss_alt_region.h5")
    ss_canonical_id_convert_path=os.path.join(ss_root,"ss_canonical_id_convert.tsv")
    #
    ds_region_fasta_path=os.path.join(ds_root,"ds_region.fasta")
    ds_rna_bed_path=os.path.join(ds_root,"ds_rna.bed")
    ds_rna_gff_path=os.path.join(ds_root,"ds_rna.gff3")
    ds_canonical_bed_path=os.path.join(ds_root,"ds_canonical.bed")
    ds_canonical_gff_path=os.path.join(ds_root,"ds_canonical.gff3")
    ###Read file###
    id_convert_dict = get_id_convert_dict(id_table_path)
    bed = read_bed(input_bed_path)
    background_bed = read_bed(background_bed_path)
    #Step 1: Remove wanted RNAs which are overlapped with unwanted data in specific distance on both strands
    recurrent_cleaned_transcripts = recurrent_cleaner(bed,background_bed,id_convert_dict,genome_path+".fai",
                                                      upstream_dist,downstream_dist,cleaning_root)
    write_bed(recurrent_cleaned_transcripts,recurrent_cleaned_transcript_path)
    recurrent_cleaned_genes = _convert_bed_id(recurrent_cleaned_transcripts,id_convert_dict)[BED_COLUMNS[:6]]
    recurrent_cleaned_genes['score'] = '.'
    recurrent_cleaned_genes = recurrent_cleaned_genes.drop_duplicates()
    write_bed(recurrent_cleaned_genes,recurrent_cleaned_gene_path)
    #Step 2: Create regions around wanted RNAs on both strand
    os.system(_SLOP_COMMAND.format(recurrent_cleaned_gene_path,genome_path+".fai",
                                   upstream_dist,downstream_dist,potential_region_path))
    potential_regions = read_bed(potential_region_path)
    potential_regions['strand'] = '+'
    potential_regions['id'] = _get_coord_id(potential_regions)
    write_bed(potential_regions,potential_region_path)
    #Step 3: Select region which its sequence only has A, T, C, and G
    _get_fasta(genome_path,potential_region_path,potential_region_fasta_path,use_name=True)
    get_cleaned_code_bed_main(potential_region_path,potential_region_fasta_path, 
                              cleaned_region_path,dirty_region_path)
    #Step 4: Get the number of region covering gene and filter the data by the number
    os.system(_GENE_TO_REGION_MAPPING_COMMAND.format(cleaned_region_path,recurrent_cleaned_gene_path,
                                                     mapping_result_path,mapping_count_path))
    if merge_overlapped:
        os.system(_SORT_MERGE_COMMAND.format(mapping_result_path,final_origin_region_bed_path))
    else:
        region_gene_count_filter_main(mapping_result_path,recurrent_cleaned_gene_path,final_origin_region_bed_path,
                                      upstream_dist,downstream_dist,discard_output_path=multi_gene_region_path)
    #Step 5: Get RNAs in selected regions
    final_gene_ids = list(read_bed(final_origin_region_bed_path,ignore_strand_check=True)['id'])
    final_transcript_ids = []
    for id_ in bed['id']:
        if id_convert_dict[id_] in final_gene_ids:
            final_transcript_ids.append(id_)
    selected_transcripts = bed[bed['id'].isin(final_transcript_ids)]
    #Step 6: Rename region
    final_origin_region_bed = read_bed(final_origin_region_bed_path,ignore_strand_check=True)
    final_origin_region_bed['id'] = '.'
    plus_strand = final_origin_region_bed.copy()
    minus_strand = final_origin_region_bed.copy()
    plus_strand['strand'],minus_strand['strand'] = '+','-'
    final_origin_strand_region_bed = pd.concat([plus_strand,minus_strand])
    write_bed(final_origin_strand_region_bed,final_origin_strand_region_bed_path)
    region_id_conversion_main(final_origin_strand_region_bed_path,region_table_path,region_bed_path)
    _get_fasta(genome_path,region_bed_path,region_fasta_path,
              use_name=True,use_strand=True)
    rename_chrom_main(region_fasta_path, region_table_path, region_fasta_path, 'coord' ,'ordinal_id_with_strand')
    _get_fai(region_fasta_path)
    #Step 7: Redefine coordinate based on double-strand region data
    region_table = read_region_table(region_table_path)
    selected_transcript_gff = bed2gff(selected_transcripts,id_convert_dict)
    ds_redefined_gff = redefine_coordinate(selected_transcript_gff,region_table)
    ds_redefined_bed = gff2bed(ds_redefined_gff)
    write_gff(ds_redefined_gff,ds_rna_gff_path)
    write_bed(ds_redefined_bed,ds_rna_bed_path)
    #Step 8: Redefine coordinate based on single-strand region data
    ss_redefined_gff = redefine_coordinate(selected_transcript_gff,region_table,with_strand=True)
    ss_redefined_bed = gff2bed(ss_redefined_gff)
    write_gff(ss_redefined_gff,rna_gff_path)
    write_bed(ss_redefined_bed,rna_bed_path)
    #Step 9: Create data about gene structure and alternative status
    create_canonical_gene_main(rna_bed_path,id_table_path,gene_gff_path=canonical_gff_path,
                               gene_bed_path=canonical_bed_path,status_gff_path=alt_region_gff_path,
                               output_id_table_path=ss_canonical_id_convert_path)
    create_canonical_gene_main(ds_rna_bed_path,id_table_path,gene_gff_path=ds_canonical_gff_path,
                               gene_bed_path=ds_canonical_bed_path)
    #Step 10: Create annotation
    create_ann_genome_main(canonical_gff_path,region_table_path,source_name,canonical_h5_path,
                           discard_alt_region=True,discard_UTR_CDS=True)
    create_ann_genome_main(alt_region_gff_path,region_table_path,source_name,alt_region_h5_path,
                           with_alt_region=True ,with_alt_site_region=True)
    ann_seqs_summary_main(canonical_h5_path,canonical_length_stats_root)
    #Step 11: Get double strand sequence
    _get_fasta(genome_path,region_bed_path,ds_region_fasta_path,use_name=True)
    rename_chrom_main(ds_region_fasta_path,region_table_path,ds_region_fasta_path,
                      source='coord',target='ordinal_id_wo_strand')
    _get_fai(ds_region_fasta_path) 
    #Step 12: Write stats
    stats = {}
    stats["1. The number of input RNAs"] = _get_bed_num(input_bed_path)
    stats["2. The number of recurrent-cleaned gene"] = _get_bed_num(recurrent_cleaned_gene_path)
    stats["3. The number of recurrent-cleaned RNAs"] = _get_bed_num(recurrent_cleaned_transcript_path)
    stats["4. The number of regions"] = _get_bed_num(potential_region_path)
    stats["5. The number of cleaned regions"] = _get_bed_num(cleaned_region_path)
    stats["6. The number of dirty regions"] = _get_bed_num(dirty_region_path)
    stats["7. The number of final regions"] = _get_bed_num(final_origin_region_bed_path)
    stats["8. The number of cleaned genes in valid regions"] = _get_bed_num(canonical_bed_path)
    stats["9. The number of cleaned RNAs in valid regions"] = _get_bed_num(rna_bed_path)
    write_json(stats,stats_path)
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_bed_path",help="Path of BED file",required=True)
    parser.add_argument("-b", "--background_bed_path",help="Path of background BED file",required=True)
    parser.add_argument("-u", "--upstream_dist", type=int,help="upstream_dist",required=True)
    parser.add_argument("-d", "--downstream_dist", type=int,help="downstream_dist",required=True)
    parser.add_argument("-g", "--genome_path",help="Path of genome fasta",required=True)
    parser.add_argument("-o", "--output_root",required=True)
    parser.add_argument("-t","--id_table_path",required=True)
    parser.add_argument("-s","--source_name",required=True)
    parser.add_argument("-m","--merge_overlapped",action='store_true')
    args = parser.parse_args()
    main(**vars(args))
