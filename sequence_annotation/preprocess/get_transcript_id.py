import os, sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict

def main(input_path, output_path, id_table_path):
    ids = set(pd.read_csv(input_path,header=None)[0])
    id_convert_dict = get_id_convert_dict(id_table_path)
    transcript_ids = []
    for transcript_id, gene_id in id_convert_dict.items():
        if gene_id in ids:
            transcript_ids.append(transcript_id)

    with open(output_path,'w') as fp:
        for id_ in transcript_ids:
            fp.write("{}\n".format(id_))

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(
        description="This program will output all transcript id in gene id list")
    parser.add_argument("-i", "--input_path",required=True,
                        help="Input gene id list file")
    parser.add_argument("-t", "--id_table_path",required=True,
                        help="Table about transcript to gene id conversion")
    parser.add_argument("-o", "--output_path",required=True,
                        help="Output transcript id list file")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
