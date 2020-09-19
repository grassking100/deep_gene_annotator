import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder
from sequence_annotation.utils.get_intersection_id import intersection
from sequence_annotation.file_process.utils import read_bed
from sequence_annotation.file_process.utils import write_bed
from sequence_annotation.file_process.get_id_table import get_id_convert_dict
from sequence_annotation.preprocess.bed_score_filter import gene_score_filter,transcript_score_filter
from sequence_annotation.preprocess.nonoverlap_filter import nonoverlap_filter
from sequence_annotation.preprocess.remove_alt_site import get_no_alt_site_transcript
from sequence_annotation.preprocess.belonging_gene_coding_filter import belonging_gene_coding_filter


def main(input_bed_path,id_table_path,output_root,non_hypothetical_protein_gene_id_path=None,
         remove_alt_site=False,remove_non_coding=False,remove_fail_score_gene=False,
         score_threshold=None,score_mode=None):
    score_mode = score_mode or "bigger_or_equal"
    # Set path
    score_filter_root=os.path.join(output_root,"score_filter")
    nonoverlap_filter_root=os.path.join(output_root,"nonoverlap_filter")
    remove_alt_site_root=os.path.join(output_root,"remove_alt_site")
    remove_non_coding_root=os.path.join(output_root,"remove_non_coding")
    passed_score_bed_path=os.path.join(score_filter_root,"passed_score_filter.bed")
    nonoverlap_bed_path=os.path.join(nonoverlap_filter_root,"nonoverlap.bed")
    overlap_bed_path = os.path.join(nonoverlap_filter_root, "overlap.bed")
    no_alt_site_bed_path=os.path.join(remove_alt_site_root,"no_alt_site.bed")
    all_coding_bed_path=os.path.join(remove_non_coding_root,"all_coding.bed")
    intersection_id_path=os.path.join(output_root,"intersection_id.txt")
    filtered_bed_path=os.path.join(output_root,"filtered.bed")
    ven_path=os.path.join(output_root,"venn.png")
    create_folder(output_root)
    create_folder(score_filter_root)
    create_folder(remove_alt_site_root)
    create_folder(remove_non_coding_root)
    
    ###Read file###
    transcript_ids_list = []
    transcript_names_list = []
    
    bed = read_bed(input_bed_path)
    id_convert_dict = get_id_convert_dict(id_table_path)
    
    nonoverlapped_bed,overlap_bed = nonoverlap_filter(bed,nonoverlap_filter_root,use_strand=True,
                                                      id_convert_dict=id_convert_dict)
    transcript_ids_list.append(nonoverlapped_bed['id'])
    transcript_names_list.append("Nonoverlapped with other gene")
    write_bed(nonoverlapped_bed, nonoverlap_bed_path)
    write_bed(overlap_bed, overlap_bed_path)
    
    if score_threshold is not None:
        if remove_fail_score_gene:
            score_bed = gene_score_filter(bed,id_convert_dict,score_threshold,score_mode)
        else:
            score_bed = transcript_score_filter(bed,score_threshold,score_mode)
        transcript_ids_list.append(score_bed['id'])
        transcript_names_list.append("Passed score")
        write_bed(score_bed,passed_score_bed_path)

    if remove_alt_site:
        no_alt_site_bed = get_no_alt_site_transcript(bed,id_convert_dict)
        transcript_ids_list.append(no_alt_site_bed['id'])
        transcript_names_list.append("No alternative splicing site")
        write_bed(no_alt_site_bed, no_alt_site_bed_path) 
    
    if remove_non_coding:
        coding_gene_transcript_bed = belonging_gene_coding_filter(bed,id_convert_dict)
        transcript_ids_list.append(coding_gene_transcript_bed['id'])
        transcript_names_list.append("All-Coding")
        write_bed(coding_gene_transcript_bed, all_coding_bed_path)
    
    if non_hypothetical_protein_gene_id_path is not None:
        real_protein_gene_ids = set(pd.read_csv(non_hypothetical_protein_gene_id_path,header=None)[0])
        real_protein_transcript_ids = []
        for transcript_id, gene_id in id_convert_dict.items():
            if gene_id in real_protein_gene_ids:
                real_protein_transcript_ids.append(transcript_id)
        transcript_ids_list.append(real_protein_transcript_ids)
        transcript_names_list.append("Non-hypothetical protein")

    intersection_ids = intersection(transcript_ids_list,transcript_names_list,
                                    output_path=intersection_id_path,
                                    venn_path=ven_path)
    filtered_bed = bed[bed['id'].isin(intersection_ids)]
    write_bed(filtered_bed, filtered_bed_path)
    

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_bed_path",help="Path of BED file",required=True)
    parser.add_argument("-o", "--output_root",required=True)
    parser.add_argument("-t","--id_table_path",required=True)
    parser.add_argument("--non_hypothetical_protein_gene_id_path")
    parser.add_argument("--score_threshold",type=float,default=None)
    parser.add_argument("--score_mode",type=str,default=None)
    parser.add_argument("--remove_alt_site",action='store_true')
    parser.add_argument("--remove_non_coding",action='store_true')
    parser.add_argument("--remove_fail_score",action='store_true')
    parser.add_argument("--remove_fail_score_gene",action='store_true')
    
    args = parser.parse_args()
    main(**vars(args))
