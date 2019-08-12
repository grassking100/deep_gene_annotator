import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import BED_COLUMNS, read_bed, write_bed
from utils import coordinate_consist_filter, create_coordinate_bed, get_id_table


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_path",
                        help="saved_root",required=True)
    parser.add_argument("-c", "--consist_data_path",
                        help="consist_data_path",required=True)
    parser.add_argument("-b", "--valid_official_bed_path",
                        help="valid_official_bed_path",required=True)
    parser.add_argument("-i", "--id_convert_path",
                        help="id_convert_path",required=True)
    parser.add_argument("-s", "--single_orf_start_end",type=lambda x: x=='true' or x=='T' or x=='t',
                        default='F',required=False)
    args = parser.parse_args()
    id_convert = get_id_table(args.id_convert_path)
    if os.path.exists(args.output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        consist_data = pd.read_csv(args.consist_data_path,sep='\t')
        valid_official_bed = read_bed(args.valid_official_bed_path)
        coordinate_consist_data = valid_official_bed.merge(consist_data,left_on='id', right_on='ref_name')
        new_data = create_coordinate_bed(consist_data,valid_official_bed)
        new_data['gene_id'] = [id_convert[id_] for id_ in new_data['id']]
        if args.single_orf_start_end:
            new_data = coordinate_consist_filter(new_data,'gene_id','thick_start')
            new_data = coordinate_consist_filter(new_data,'gene_id','thick_end')
        new_data = new_data[BED_COLUMNS]
        coord_info = new_data[['chr','start','end','strand','thick_start','thick_end',
                               'count','block_size','block_related_start']]
        new_data = new_data[~ coord_info.duplicated()].sort_values(by=['chr','start','end','strand'])
        write_bed(new_data,args.output_path)
