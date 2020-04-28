import os,sys
import pandas as pd
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict

def get_subbed(bed,ids,id_table_path=None,query_column=None):
    query_column = query_column or 'id'
    id_convert_dict= None
    if id_table_path is not None:
        id_convert_dict = get_id_convert_dict(id_table_path)

    bed = bed.to_dict('record')
    new_bed = []
    for item in bed:
        query_id = item[query_column]
        if id_convert_dict is not None:
            #Convert to id based on table
            query_id = id_convert_dict[query_id]
        if query_id in ids:
            new_bed.append(item)

    new_bed = pd.DataFrame.from_dict(new_bed)
    return new_bed
    
def main(input_path,id_path,output_path,treat_id_path_as_ids=False,**kwargs):
    bed = read_bed(input_path)
    if treat_id_path_as_ids:
        ids = id_path.split(',')
    else:
        try:
            data = set(list(pd.read_csv(id_path,header=None,sep='\t')[0]))
            ids = []
            for item in data:
                ids += item.split(',')

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
    parser.add_argument("-t", "--id_table_path")
    parser.add_argument("--query_column",type=str,help="Defualt :id")
    parser.add_argument("--treat_id_path_as_ids",action='store_true')
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
