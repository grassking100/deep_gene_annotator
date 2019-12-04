import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed, write_bed

def both_strand_include_filter(bed):
    bed['coord_id'] = bed['chr'].astype(str)+bed['start'].astype(str)+bed['end'].astype(str)
    coord_ids = set(list(bed['coord_id']))
    returned = []
    discarded = []
    for coord_id in coord_ids:
        selected = bed[bed['coord_id']==coord_id]
        if len(selected)==2:
            returned += selected.to_dict('record')
        else:
            discarded += selected.to_dict('record')
    return pd.DataFrame.from_dict(returned),pd.DataFrame.from_dict(discarded)
    
def main(bed_path,output_path,discarded_path):
    bed = read_bed(bed_path)
    returned,discarded = both_strand_include_filter(bed)
    write_bed(returned,output_path)
    if args.discarded_path is not None:
        write_bed(discarded,discarded_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will preserve region which have other strand in data")
    parser.add_argument("-i", "--bed_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("-d", "--discarded_path")
    args = parser.parse_args()
    main(args.bed_path,args.output_path,args.discarded_path)
