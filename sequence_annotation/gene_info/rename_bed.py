import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed

def rename_bed(bed,use_strand,use_id_as_id,id_prefix):
    bed = bed.sort_values(by=['chr','start','end','strand']).to_dict('record')
    renamed_table = []
    renamed_bed = []
    index_length = len(str(len(bed)))
    seq_id = "{prefix}_{index:0>"+str(index_length)+"d}"
    id_table = {}
    index = 1
    columns = ['chr','start','end']
    if use_strand:
        columns.append('strand')
    for item in bed:
        coord_id = '_'.join([str(item[key]) for key in columns])
        new_id = seq_id.format(prefix=id_prefix,index=index)    
        if coord_id not in id_table.keys():
            id_table[coord_id] = new_id
            index += 1

    for item in bed:
        old_id = item['id']
        coord_id = '_'.join([str(item[key]) for key in columns])
        table_item = {}
        for key in ['chr','strand','start','end']:
            table_item[key] = item[key]
        table_item['new_id'] = id_table[coord_id]
        if use_id_as_id:
            table_item['old_id'] = old_id
        else:
            table_item['old_id'] = coord_id
        renamed_table.append(table_item)
        renamed_item = dict(item)
        renamed_item['id'] = id_table[coord_id]
        renamed_bed += [renamed_item]

    renamed_table = pd.DataFrame.from_dict(renamed_table).sort_values(by=['new_id'])
    renamed_bed = pd.DataFrame.from_dict(renamed_bed).sort_values(by=['id'])
    renamed_table = renamed_table[['new_id','old_id','chr','strand','start','end']]
    return renamed_table,renamed_bed

def main(bed_path,use_strand,use_id_as_id,id_prefix,saved_table_path,renamed_bed_path):
    try:
        bed = read_bed(bed_path)
    except:
        raise Exception("Wrong format in {}".format(bed_path))
    renamed_table,renamed_bed = rename_bed(bed,use_strand,use_id_as_id,id_prefix)
    renamed_table.to_csv(saved_table_path,index=None,sep='\t')
    write_bed(renamed_bed,renamed_bed_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will rename bed id field by cooridnate data")
    parser.add_argument("-i", "--bed_path",help="Bed file to be renamed",required=True)
    parser.add_argument("-p", "--id_prefix",help="Prefix of new id",required=True)
    parser.add_argument("-t", "--saved_table_path",help="Path to saved one-based renamed table",required=True)
    parser.add_argument("-o", "--renamed_bed_path",help="Path to saved renamed bed file",required=True)
    parser.add_argument("-s", "--use_strand",action='store_true')
    parser.add_argument("--use_id_as_id",action='store_true',
                        help='Use id as renamed table\'s old_id, otherwise use cooridnate data.')
    
    args = parser.parse_args()
     
    main(args.bed_path,args.use_strand,args.use_id_as_id,args.id_prefix,
         args.saved_table_path,args.renamed_bed_path)
