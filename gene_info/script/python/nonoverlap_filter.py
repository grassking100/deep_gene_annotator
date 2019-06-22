import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import get_id_table, read_bed, write_bed, simply_coord_with_gene_id
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    root_path = "/".join(sys.argv[0].split('/')[:-1])
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root",help="saved_root",required=True)
    parser.add_argument("-i", "--id_convert_path",help="id_convert_path",required=False)
    parser.add_argument("-c", "--coordinate_consist_bed_path",help="coordinate_consist_bed_path",required=True)
    args = vars(parser.parse_args())
    saved_root = args['saved_root']
    output_path = os.path.join(saved_root,"nonoverlap.bed")
    overlap_path = os.path.join(saved_root,"overlap.bed")
    gene_bed_path = os.path.join(saved_root,"gene_coord.bed")
    nonoverlap_id_path = os.path.join(saved_root,"gene_coord_nonoverlap_id.txt")
    if os.path.exists(output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        ###Read data###
        id_convert = None
        if args.id_convert_path is not None:
            id_convert = get_id_table(args.id_convert_path)
        coordinate_consist_bed = read_bed(args.coordinate_consist_bed_path)
        gene_bed = simply_coord_with_gene_id(coordinate_consist_bed,id_convert)
        write_bed(gene_bed,gene_bed_path)
        command = "bash {}/../bash/nonoverlap_filter.sh {}".format(root_path,gene_bed_path)
        print("Execute "+command)
        os.system(command)
        nonoverlap_id = [id_ for id_ in open(nonoverlap_id_path).read().split('\n') if id_ != '']
        ###Write data###
        if id_convert is not None:
            coordinate_consist_bed['gene_id'] = [id_convert[id_] for id_ in coordinate_consist_bed['id']]
        else:
            coordinate_consist_bed['gene_id'] = coordinate_consist_bed['id']
        want_bed = coordinate_consist_bed[coordinate_consist_bed['gene_id'].isin(nonoverlap_id)]
        want_bed = want_bed[~ want_bed.duplicated()]
        write_bed(want_bed,output_path)
        
        overlap_bed = coordinate_consist_bed[~coordinate_consist_bed['gene_id'].isin(nonoverlap_id)]
        overlap_bed = overlap_bed[~ overlap_bed.duplicated()]
        write_bed(overlap_bed,overlap_path)
