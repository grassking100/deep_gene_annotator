import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import get_id_table, read_bed, write_bed, simply_coord_with_gene_id
import pandas as pd
from argparse import ArgumentParser
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root",help="saved_root",required=True)
    parser.add_argument("-i", "--id_convert_path",help="id_convert_path",required=True)
    parser.add_argument("-c", "--coordinate_consist_bed_path",help="coordinate_consist_bed_path",required=True)
    args = vars(parser.parse_args())
    saved_root = args['saved_root']
    id_convert_path = args['id_convert_path']
    coordinate_consist_bed_path = args['coordinate_consist_bed_path']
    output_path = saved_root+"/nonoverlap.bed"
    gene_bed_path = saved_root+"/gene_coord.bed"
    nonoverlap_id_path = saved_root+"/gene_coord_nonoverlap_id.txt"
    if os.path.exists(output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        ###Read data###
        id_convert = get_id_table(id_convert_path)
        coordinate_consist_bed = read_bed(coordinate_consist_bed_path)
        gene_bed = simply_coord_with_gene_id(coordinate_consist_bed,id_convert)
        write_bed(gene_bed,gene_bed_path)
        command = "bash ./sequence_annotation/gene_info/script/bash/nonoverlap_filter.sh "+ gene_bed_path
        os.system(command)
        nonoverlap_id = [id_ for id_ in open(nonoverlap_id_path).read().split('\n') if id_ != '']
        ###Write data###
        coordinate_consist_bed['gene_id'] = [id_convert[id_] for id_ in coordinate_consist_bed['id']]
        nonoverlap_id
        want_bed = coordinate_consist_bed[coordinate_consist_bed['gene_id'].isin(nonoverlap_id)]
        want_bed = want_bed[~ want_bed.duplicated()]
        write_bed(want_bed,output_path)
        