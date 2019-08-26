import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will redefine coordinate based on region file")
    parser.add_argument("-i", "--bed_path",help="Bed file to be redefined coordinate",required=True)
    parser.add_argument("-t", "--table_path",help="Table about renamed region",required=True)
    parser.add_argument("-r", "--region_path",help="Region file to be used as coordinate reference",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined bed file",required=True)
    parser.add_argument("-f", "--flip",help="Flip sequence info which are minus strand to plus strand",
                        action="store_true", required=False)
    args = parser.parse_args()
    bed = read_bed(args.bed_path).to_dict('record')
    region = read_bed(args.region_path).to_dict('record')
    table = pd.read_csv(args.table_path,sep='\t').to_dict('record')
    starts = dict()
    ends = dict()
    lengths = dict()
    for item in region:
        id_ = item['id']
        starts[id_] = item['start']
        ends[id_] = item['end']
        lengths[id_] = item['end'] - item['start'] + 1
        
    belong_table = {}
    for item in table:
        ids = item['old_id'].split(",")
        for id_ in ids:
            belong_table[id_] = item['new_id']
    
    redefined_bed = []
    for index,item in enumerate(bed):
        bed_item = dict(item)
        belong_id = belong_table[item['id']]
        if not args.flip:
            anchor = starts[belong_id] - 1
            bed_item['start'] -= anchor
            bed_item['end'] -= anchor
            bed_item['thick_start'] -= anchor
            bed_item['thick_end'] -= anchor
        else:    
            if bed_item['strand'] == '+':    
                anchor = starts[belong_id] - 1
                bed_item['start'] -= anchor
                bed_item['end'] -= anchor
                bed_item['thick_start'] -= anchor
                bed_item['thick_end'] -= anchor
            else:
                anchor = ends[belong_id] + 1
                bed_item['strand'] = '+'
                length = lengths[belong_id]
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
        bed_item['chr'] = belong_id
        redefined_bed += [bed_item]

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    write_bed(redefined_bed,args.output_path)