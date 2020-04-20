import os, sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_bed, write_bed
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict


def convert_bed_id(bed, id_convert_dict, query=None):
    query = query or 'id'
    returned = []
    for item in bed.to_dict('record'):
        item = dict(item)
        item[query] = id_convert_dict[item[query]]
        returned.append(item)
    returned = pd.DataFrame.from_dict(returned).sort_values(by=['id'])
    return returned


def main(input_bed_path, output_bed_path, id_table_path, query=None):
    bed = read_bed(input_bed_path)
    try:
        id_convert_dict = get_id_convert_dict(id_table_path)
    except:
        raise Exception("Something wrong happened in {}".format(
            args.id_table_path))
    returned = convert_bed_id(bed, id_convert_dict, query)
    write_bed(returned, output_bed_path)


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(
        description="This program will rename query data by table")
    parser.add_argument("-i", "--input_bed_path",required=True,
                        help="Input BED file")
    parser.add_argument("-t", "--id_table_path",required=True,
                        help="Table about id conversion")
    parser.add_argument("-o", "--output_bed_path",required=True,
                        help="Output BED file")
    parser.add_argument("--query", default='id',
                        help="Column name to query and replace")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
