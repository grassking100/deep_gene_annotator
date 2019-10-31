import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import BED_COLUMNS, read_bed, write_bed
from utils import get_id_table, coordinate_consist_filter 

def create_coordinate_bed(consist_data,bed):
    consist_data['start_diff'] = consist_data['coordinate_start'] - consist_data['start']
    consist_data['end_diff'] = consist_data['coordinate_end'] - consist_data['end']
    returned = []
    for item in consist_data.to_dict('record'):
        count = int(item['count'])
        block_related_start = [int(val) for val in item['block_related_start'].split(',')[:count]]
        block_size = [int(val) for val in item['block_size'].split(',')[:count]]
        start_diff = int(item['start_diff'])
        end_diff = int(item['end_diff'])
        block_related_start = [start-start_diff for start in block_related_start]
        block_related_start[0] = 0
        block_size[0] -= start_diff
        block_size[-1] += end_diff
        for val in block_related_start:
            if val<0:
                raise Exception(item['ref_name']+" has negative relative start site,"+str(val))
        for val in block_size:
            if val<=0:
                raise Exception(item['ref_name']+" has nonpositive size,"+str(val))
        template = dict(item)
        template['block_related_start'] = ','.join(str(c) for c in block_related_start)
        template['block_size'] = ','.join(str(c) for c in block_size)
        template['start'] = template['coordinate_start']
        template['end'] = template['coordinate_end']
        returned.append(template)
    return pd.DataFrame.from_dict(returned)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will create data in BED format"
                            " by cooridnate data and origin data in BED foramt")
    parser.add_argument("-i", "--bed_path",required=True)
    parser.add_argument("-c", "--coordinate_data_path",required=True)
    parser.add_argument("-t", "--id_convert_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("--single_orf_start_end",action='store_true',
                        help="If it is selected, then only gene data "+
                        "which have single ORF will be saved")
    args = parser.parse_args()
    id_convert = get_id_table(args.id_convert_path)
    if os.path.exists(args.output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        coordinat_data = pd.read_csv(args.coordinate_data_path,sep='\t')
        bed = read_bed(args.bed_path)
        coordinat_data = bed.merge(coordinat_data,left_on='id', right_on='ref_name')
        coordinate_bed = create_coordinate_bed(coordinat_data,bed)
        coordinate_bed['gene_id'] = [id_convert[id_] for id_ in coordinate_bed['id']]
        if args.single_orf_start_end:
            coordinate_bed = coordinate_consist_filter(coordinate_bed,'gene_id','thick_start')
            coordinate_bed = coordinate_consist_filter(coordinate_bed,'gene_id','thick_end')
        coordinate_bed = coordinate_bed[BED_COLUMNS]
        coordinate_bed = coordinate_bed[~coordinate_bed.duplicated()].sort_values(by=['chr','start','end','strand'])
        write_bed(coordinate_bed,args.output_path)