import os,sys
import pandas as pd
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed
from sequence_annotation.preprocess.utils import get_id_table

def get_subbed(bed,ids,id_convert_path=None,query_column=None):
    query_column = query_column or 'id'
    id_table = None
    if id_convert_path is not None:
        id_table = get_id_table(id_convert_path)

    bed = bed.to_dict('record')
    new_bed = []
    for item in bed:
        query_id = item[query_column]
        if id_table is not None:
            #Convert to id based on table
            query_id = id_table[query_id]
        if query_id in ids:
            new_bed.append(item)

    new_bed = pd.DataFrame.from_dict(new_bed)
    return new_bed
    
def main(input_path,id_path,output_path,**kwargs):
    bed = read_bed(input_path)
    try:
        ids = set(list(pd.read_csv(id_path,header=None)[0]))
    except:
        raise Exception("The path {} is not exist".format(id_path))
    part_bed = get_subbed(bed,ids,**kwargs)
    write_bed(part_bed,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",required=True)
    parser.add_argument("-d", "--id_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("-t", "--id_convert_path")
    parser.add_argument("--query_column",type=str)
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
