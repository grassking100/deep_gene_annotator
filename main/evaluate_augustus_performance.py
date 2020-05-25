import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/..")
from sequence_annotation.utils.utils import read_gff, write_gff,create_folder
from sequence_annotation.preprocess.utils import read_region_table
from sequence_annotation.preprocess.create_gene_with_alt_status_gff import convert_from_gff_to_gene_with_alt_status_gff
from sequence_annotation.preprocess.create_canonical_gene import create_gene_gff_from_gene_alt_status_gff
from sequence_annotation.process.performance import compare_and_save
    

def compare_gff_gene_result(predicted_path,answer_path,region_table_path,
                            evaluate_root,output_gene_path):
    predicted = read_gff(predicted_path)
    answer = read_gff(answer_path)
    region_table = read_region_table(region_table_path)
    statuses = convert_from_gff_to_gene_with_alt_status_gff(predicted,
                                                            select_site_by_election=True,
                                                            allow_partial_gene=True)
    predicted_gene = create_gene_gff_from_gene_alt_status_gff(statuses)
    write_gff(predicted_gene,output_gene_path)
    compare_and_save(predicted_gene,answer,region_table,evaluate_root)

def main(augustus_result_root,train_val_answer_path,test_answer_path,
         region_table_path,output_root):
    
    #Set root
    create_folder(output_root)
    train_root = os.path.join(augustus_result_root,'train')
    test_root = os.path.join(augustus_result_root,'test')
    train_evaluate_root = os.path.join(output_root,'train')
    test_evaluate_root = os.path.join(output_root,'test')
    #Set path
    train_predicted_path = os.path.join(train_root,'train.final.predict.gff')
    test_predicted_path = os.path.join(test_root,'test.final.predict.gff')
    train_gene_path = os.path.join(train_evaluate_root,'train.final.predict_gene.gff')
    test_gene_path = os.path.join(test_evaluate_root,'test.final.predict_gene.gff')
    
    compare_gff_gene_result(train_predicted_path,train_val_answer_path,region_table_path,
                            train_evaluate_root,train_gene_path)
    
    compare_gff_gene_result(test_predicted_path,test_answer_path,region_table_path,
                            test_evaluate_root,test_gene_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-a","--augustus_result_root",help="Root of augustus training file",required=True)
    parser.add_argument("-t","--train_val_answer_path",help='The training and validation answer in gff format',required=True)
    parser.add_argument("-x","--test_answer_path",help='The testing answer in gff format',required=True)
    parser.add_argument("-r","--region_table_path",help="The path of region data table",required=True)
    parser.add_argument("-o","--output_root",required=True)
    
    args = parser.parse_args()
    main(**vars(args))
