import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed

def redefine_coordinate(bed,renamed_table):
    belong_table = {}
    for item in renamed_table.to_dict('record'):
        id_ = item['old_id']
        belong_table[id_] = item['new_id']
    redefined_bed = []
    for old_id in belong_table.keys():
        old_id
    for item in bed.to_dict('record'):
        match_chr = renamed_table['chr'] == item['chr']
        selected = renamed_table[match_chr]
        if len(selected) < 1:
            raise Exception("Cannot locate {}".format(item['id']))
        for region in selected.to_dict('record'):
            bed_item = dict(item)
            if bed_item['strand'] == '-':
                anchor = region['end'] + 1
                length = region['end'] - region['start'] + 1
                block_starts = [int(v)+bed_item['start'] for v in  bed_item['block_related_start'].split(",")]
                block_sizes = [v for v in  bed_item['block_size'].split(",")]
                block_ends = [block_start + int(block_size) -1 for block_start,block_size in zip(block_starts,block_sizes)]
                #Start recoordinate
                start = bed_item['start']
                end = bed_item['end']
                thick_start = bed_item['thick_start']
                thick_end = bed_item['thick_end']
                bed_item['end'] = anchor - start
                bed_item['start'] = anchor - end
                bed_item['thick_end'] = anchor - thick_start
                bed_item['thick_start'] = anchor - thick_end
                new_block_starts = [str(anchor - block_end - bed_item['start']) for block_end in block_ends]
                bed_item['block_related_start'] = ','.join(reversed(new_block_starts))
                bed_item['block_size'] = ','.join(reversed(block_sizes))
            bed_item['chr'] = region['new_id']
            redefined_bed += [bed_item]

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    return redefined_bed

def main(bed_path,table_path,output_path):
    bed = read_bed(bed_path)
    renamed_table = pd.read_csv(table_path,sep='\t',dtype={'chr':str,'start':int,'end':int})
    redefined_bed = redefine_coordinate(bed,renamed_table)
    write_bed(redefined_bed,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will merge two strand regions into one strand regions, the coordinate data of flipped minus strand would be flipped back")
    parser.add_argument("-i", "--bed_path",help="Bed file to be redefined coordinate",required=True)
    parser.add_argument("-t", "--table_path",help="Table about renamed region",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined bed file",required=True)
    
    args = parser.parse_args()
    main(args.bed_path,args.table_path,args.output_path)
    