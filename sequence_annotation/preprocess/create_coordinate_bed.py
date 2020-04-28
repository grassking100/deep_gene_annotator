import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_gff, get_gff_with_attribute
from sequence_annotation.utils.utils import read_bed, write_bed
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict
from sequence_annotation.preprocess.create_coordinate_data import coordinate_consist_filter


def create_coordinate_bed(coordinate):
    coordinate['start_diff'] = coordinate['coordinate_start'] - coordinate['start']
    coordinate['end_diff'] = coordinate['coordinate_end'] - coordinate['end']
    returned = []
    for item in coordinate.to_dict('record'):
        count = int(item['count'])
        block_related_start = [
            int(val) for val in item['block_related_start'].split(',')[:count]
        ]
        block_size = [
            int(val) for val in item['block_size'].split(',')[:count]
        ]
        start_diff = int(item['start_diff'])
        end_diff = int(item['end_diff'])
        block_related_start = [
            start - start_diff for start in block_related_start
        ]
        block_related_start[0] = 0
        block_size[0] -= start_diff
        block_size[-1] += end_diff
        for val in block_related_start:
            if val < 0:
                raise Exception(item['ref_name'] +
                                " has negative relative start site," +
                                str(val))
        for val in block_size:
            if val <= 0:
                raise Exception(item['ref_name'] + " has nonpositive size," +
                                str(val))
        template = dict(item)
        template['block_related_start'] = ','.join(
            str(c) for c in block_related_start)
        template['block_size'] = ','.join(str(c) for c in block_size)
        template['start'] = template['coordinate_start']
        template['end'] = template['coordinate_end']
        returned.append(template)
    return pd.DataFrame.from_dict(returned)


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will create "
                            "data in BED format by cooridnate data "
                            "and origin data in BED foramt")
    parser.add_argument("-i", "--bed_path", required=True)
    parser.add_argument("-c", "--coordinate_data_path", required=True)
    parser.add_argument("-t", "--id_table_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("--single_orf_start_end", action='store_true',
                        help="If it is selected, then only gene data "
                        "which have single ORF will be saved")
    args = parser.parse_args()

    id_convert_dict = get_id_convert_dict(args.id_table_path)

    if os.path.exists(args.output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        coordinate_data = get_gff_with_attribute(
            read_gff(args.coordinate_data_path))
        coordinate_data = coordinate_data[['start', 'end', 'ref_name']]
        coordinate_data = coordinate_data.rename(columns={
            'start': 'coordinate_start',
            'end': 'coordinate_end'
        })
        bed = read_bed(args.bed_path)
        coordinate_bed_data = bed.merge(coordinate_data,
                                        left_on='id',
                                        right_on='ref_name')
        bed = create_coordinate_bed(coordinate_bed_data)
        bed['gene_id'] = [id_convert_dict[id_] for id_ in bed['id']]
        if args.single_orf_start_end:
            bed = coordinate_consist_filter(bed,'gene_id','thick_start')
            bed = coordinate_consist_filter(bed,'gene_id', 'thick_end')
        bed = bed.sort_values(by=['chr', 'start','end', 'strand'])
        write_bed(bed, args.output_path)
