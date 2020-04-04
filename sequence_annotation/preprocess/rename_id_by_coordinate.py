import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed
from sequence_annotation.preprocess.convert_bed_id import convert_bed_id

def get_coordinate_id_table(bed,query_columns,id_prefix):
    bed = bed.sort_values(by=['chr','start','end','strand']).to_dict('record')
    index_length = len(str(len(bed)))
    seq_id = "{prefix}_{index:0>"+str(index_length)+"d}"
    coord_id_table = {}
    index = 1
    for item in bed:
        coord_id = '_'.join([str(item[key]) for key in query_columns])
        new_id = seq_id.format(prefix=id_prefix,index=index)    
        if coord_id not in coord_id_table.keys():
            coord_id_table[coord_id] = new_id
            index += 1
    return coord_id_table

def get_region_table(bed,coord_id_table,query_columns,coord_id_as_old_id):
    table = []
    for item in bed.to_dict('record'):
        old_id = item['id']
        coord_id = '_'.join([str(item[key]) for key in query_columns])
        table_item = {}
        for key in ['chr','strand','start','end']:
            table_item[key] = item[key]
        table_item['new_id'] = coord_id_table[coord_id]
        if coord_id_as_old_id:
            table_item['old_id'] = coord_id
        else:
            table_item['old_id'] = old_id
        table.append(table_item)

    table = pd.DataFrame.from_dict(table).sort_values(by=['new_id'])
    table = table[['new_id','old_id','chr','strand','start','end']]
    return table    

def main(bed_path,use_strand,coord_id_as_old_id,id_prefix,
         saved_table_path,renamed_bed_path,ignore_output_strand=False):
    try:
        bed = read_bed(bed_path)
    except:
        raise Exception("Wrong format in {}".format(bed_path))
        
    query_columns = ['chr','start','end']
    if use_strand:
        query_columns.append('strand')
        
    coord_id_table = get_coordinate_id_table(bed,query_columns,id_prefix)
    region_table = get_region_table(bed,coord_id_table,query_columns,coord_id_as_old_id)
    bed['id'] = bed[query_columns].apply(lambda x: '_'.join([str(item) for item in x]), axis=1)
    if not ignore_output_strand:
        bed['strand'] = '.'    
    bed = bed.drop_duplicates()
    renamed_bed = convert_bed_id(bed,coord_id_table,'id').drop_duplicates()
    region_table.to_csv(saved_table_path,index=None,sep='\t')
    write_bed(renamed_bed,renamed_bed_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will rename bed id field by cooridnate data")
    parser.add_argument("-i", "--bed_path",help="Bed file to be renamed",required=True)
    parser.add_argument("-p", "--id_prefix",help="Prefix of new id",required=True)
    parser.add_argument("-t", "--saved_table_path",help="Path to saved one-based renamed table",required=True)
    parser.add_argument("-o", "--renamed_bed_path",help="Path to saved renamed bed file",required=True)
    parser.add_argument("-s", "--use_strand",action='store_true')
    parser.add_argument("--coord_id_as_old_id",action='store_true',
                        help='Use cooridnate id as renamed table\'s old_id, otherwise use id.')
    parser.add_argument("--ignore_output_strand",action='store_true')
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
