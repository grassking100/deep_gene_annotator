import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed, write_bed


def remove_coordinate_duplicated_bed(bed):
    bed['coord_id'] = bed['chr'].astype(str) + bed['start'].astype(
        str) + bed['end'].astype(str) + bed['strand'].astype(str)
    coord_ids = set(list(bed['coord_id']))
    returned = []
    for coord_id in coord_ids:
        selected = bed[bed['coord_id'] == coord_id]
        ids = ','.join(set(list(selected['id'])))
        selected_ = selected.to_dict('record')
        temp = selected_[0]
        temp['score'] = '.'
        temp['id'] = ids
        returned.append(temp)
    return pd.DataFrame.from_dict(returned)


def main(bed_path, output_path):
    bed = read_bed(bed_path)
    returned = remove_coordinate_duplicated_bed(bed)
    write_bed(returned, output_path)


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(
        description=
        "This program will remove duplicated coordinate data, but preserve all id"
    )
    parser.add_argument("-i", "--bed_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()
    main(args.bed_path, args.output_path)
