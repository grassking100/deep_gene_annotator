import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import read_gff,write_gff,read_region_table,get_gff_with_attribute
from sequence_annotation.preprocess.convert_transcript_to_gene_with_alt_status_gff import convert_transcript_gff_to_gene_with_alt_status_gff
from sequence_annotation.preprocess.create_gene_bed_from_exon_gff import create_gene_bed_from_exon_gff
from sequence_annotation.preprocess.bed2gff import bed2gff
from sequence_annotation.preprocess.get_id_table import get_id_table,convert_id_table_to_dict
from sequence_annotation.process.performance import compare_and_save
    
def create_gene_gff_from_gene_alt_status_gff(gene_with_alt_status_gff):
    id_table = get_id_table(get_gff_with_attribute(gene_with_alt_status_gff))
    id_convert_dict = convert_id_table_to_dict(id_table)
    gene_bed = create_gene_bed_from_exon_gff(gene_with_alt_status_gff)
    gene_gff = bed2gff(gene_bed,id_convert_dict)
    return gene_gff
    
def compare_gff_gene_result(predicted_path,answer_path,region_table_path,
                            evaluate_root,output_alt_status_path,chrom_target):
    predicted = read_gff(predicted_path)
    answer = read_gff(answer_path)
    region_table = read_region_table(region_table_path)
    predicted_gene_with_alt_status = convert_transcript_gff_to_gene_with_alt_status_gff(predicted,
                                                                                       select_site_by_election=True,
                                                                                       allow_partial_gene=True)
    write_gff(predicted_gene_with_alt_status,output_alt_status_path)
    predicted_gene = create_gene_gff_from_gene_alt_status_gff(predicted_gene_with_alt_status)
    compare_and_save(predicted_gene,answer,region_table,evaluate_root,chrom_target)

def main(augustus_result_root,train_val_answer_path,test_answer_path,
         region_table_path,is_answer_double_strand):
    
    if is_answer_double_strand:
        chrom_target = 'new_id'
    else:
        chrom_target = 'old_id'

    #Set root
    train_root = os.path.join(augustus_result_root,'train')
    test_root = os.path.join(augustus_result_root,'test')
    train_evaluate_root = os.path.join(train_root,'evaluate')
    test_evaluate_root = os.path.join(test_root,'evaluate')
    #Set path
    train_predicted_path = os.path.join(train_root,'train.final.predict.gff3')
    test_predicted_path = os.path.join(test_root,'test.final.predict.gff3')
    train_gene_with_alt_status_path = os.path.join(train_root,'train.final.predict_gene_with_alt_status.gff3')
    test_gene_with_alt_status_path = os.path.join(test_root,'test.final.predict_gene_with_alt_status.gff3')
    
    compare_gff_gene_result(train_predicted_path,train_val_answer_path,region_table_path,
                            train_evaluate_root,train_gene_with_alt_status_path,chrom_target)
    
    compare_gff_gene_result(test_predicted_path,test_answer_path,region_table_path,
                            test_evaluate_root,test_gene_with_alt_status_path,chrom_target)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-a","--augustus_result_root",help="Root of augustus training file",required=True)
    parser.add_argument("-t","--train_val_answer_path",help='The training and validation answer in gff format',required=True)
    parser.add_argument("-x","--test_answer_path",help='The testing answer in gff format',required=True)
    parser.add_argument("-r","--region_table_path",help="The path of region data table which its old_id is single-strand data's "\
                        "chromosome and new_id is double-strand data's chromosome",required=True)
    parser.add_argument("--is_answer_double_strand",action="store_true",required=True)
    
    args = parser.parse_args()
    main(**vars(args))
