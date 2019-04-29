import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import coordinate_consist_filter, create_coordinate_bed, read_bed, BED_COLUMNS, write_bed, get_id_table
import os
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root",
                        help="saved_root",required=True)
    parser.add_argument("-c", "--consist_data_path",
                        help="consist_data_path",required=True)
    parser.add_argument("-o", "--valid_official_bed_path",
                        help="valid_official_bed_path",required=True)
    parser.add_argument("-i", "--id_convert_path",
                        help="id_convert_path",required=True)
    args = vars(parser.parse_args())
    saved_root = args['saved_root']
    consist_data_path = args['consist_data_path']
    valid_official_bed_path = args['valid_official_bed_path']
    id_convert_path = args['id_convert_path']
    id_convert = get_id_table(id_convert_path)
    coordinate_consist_bed_path = saved_root+'/coordinate_consist.bed'
    if os.path.exists(coordinate_consist_bed_path):
        print("Result files are already exist,procedure will be skipped.")
    else:
        consist_data = pd.read_csv(consist_data_path,sep='\t')
        valid_official_bed = read_bed(valid_official_bed_path)

        coordinate_consist_data = valid_official_bed.merge(consist_data,left_on='id', right_on='ref_name')
        new_data = create_coordinate_bed(consist_data,valid_official_bed)
        new_data['gene_id'] = [id_convert[id_] for id_ in new_data['id']]
        #new_data = coordinate_consist_filter(new_data,'gene_id','orf_start')
        #new_data = coordinate_consist_filter(new_data,'gene_id','orf_end')
        new_data = new_data[BED_COLUMNS]
        new_data = new_data[~ new_data.duplicated()]
        write_bed(new_data,coordinate_consist_bed_path)
