import os,sys
import pandas as pd
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed
from sequence_annotation.preprocess.utils import get_id_table

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",required=True)
    parser.add_argument("-d", "--id_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("-t", "--id_convert_path")
    parser.add_argument("--exclude_with_id",action='store_true',
                        help='If it is true, then matched data would be output, otherwirse unmatched data would be output')
    parser.add_argument("--query_column",type=str,default='id')
    
    args = parser.parse_args()
    id_table = None
    if args.id_convert_path is not None:
        id_table = get_id_table(args.id_convert_path)
        
    try:
        ids = set(list(pd.read_csv(args.id_path,header=None)[0]))
    except:
        raise Exception("The path {} is not exist".format(args.id_path))

    bed = read_bed(args.input_path).to_dict('record')
    new_bed = []
    for item in bed:
        query_id = item[args.query_column]
        if id_table is not None:
            #Convert to id based on table
            query_id = id_table[query_id]
        if not args.exclude_with_id:
            if query_id in ids:
                new_bed.append(item)
        else:
            if query_id not in ids:
                new_bed.append(item)

    new_bed = pd.DataFrame.from_dict(new_bed)
    write_bed(new_bed,args.output_path)
