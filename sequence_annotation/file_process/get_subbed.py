import os,sys
import pandas as pd
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.file_process.utils import read_bed,write_bed
from sequence_annotation.file_process.get_id_table import get_id_convert_dict

def get_subbed(bed,ids,id_convert_dict=None,query_column=None,convert_input_id=False):
    ids = list(ids)
    query_column = query_column or 'id'
    if id_convert_dict is not None and convert_input_id:
        new_ids = []
        for id_ in ids:
            new_ids.append(id_convert_dict[id_])
        ids = new_ids

    bed = bed.to_dict('record')
    new_bed = []
    for item in bed:
        query_id = item[query_column]
        if id_convert_dict is not None and not convert_input_id:
            #Convert to id based on table
            query_id = id_convert_dict[query_id]
        if query_id in ids:
            new_bed.append(item)

    new_bed = pd.DataFrame.from_dict(new_bed)
    return new_bed
    
def main(input_path,id_path,output_path,id_convert_table_path=None,treat_id_path_as_ids=False,**kwargs):
    bed = read_bed(input_path)
    id_convert_dict = get_id_convert_dict(id_convert_table_path)
    if treat_id_path_as_ids:
        ids = id_path.split(',')
    else:
        data = set(list(pd.read_csv(id_path,header=None,sep='\t')[0]))
        ids = []
        for item in data:
            ids += item.split(',')
    part_bed = get_subbed(bed,ids,id_convert_dict=id_convert_dict,**kwargs)
    write_bed(part_bed,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",required=True)
    parser.add_argument("-d", "--id_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("-t", "--id_convert_table_path")
    parser.add_argument("--query_column",type=str,help="Defualt :id")
    parser.add_argument("--treat_id_path_as_ids",action='store_true')
    parser.add_argument("--convert_input_id",action='store_true')
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
