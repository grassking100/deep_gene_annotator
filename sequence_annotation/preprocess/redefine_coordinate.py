import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed

def redefine_coordinate(bed,renamed_table,use_strand,flip):
    belong_table = {}
    for item in renamed_table.to_dict('record'):
        id_ = item['old_id']
        belong_table[id_] = item['new_id']
    redefined_bed = []
    for item in bed.to_dict('record'):
        match_strand = renamed_table['strand'] == item['strand']
        match_chr = renamed_table['chr'] == item['chr']
        match_start = renamed_table['start'] <= item['start']
        match_end = renamed_table['end'] >= item['end']
        if use_strand:
            selected = renamed_table[(match_strand) & (match_chr) & (match_start) & (match_end)]
        else:    
            selected = renamed_table[(match_chr) & (match_start) & (match_end)]
        if len(selected) < 1:
            raise Exception("Cannot locate {}".format(item['id']))
        for region in selected.to_dict('record'):
            bed_item = dict(item)
            if not flip:
                anchor = region['start'] - 1
                bed_item['start'] -= anchor
                bed_item['end'] -= anchor
                bed_item['thick_start'] -= anchor
                bed_item['thick_end'] -= anchor
            else:    
                if bed_item['strand'] == '+':    
                    anchor = region['start'] - 1
                    bed_item['start'] -= anchor
                    bed_item['end'] -= anchor
                    bed_item['thick_start'] -= anchor
                    bed_item['thick_end'] -= anchor
                else:
                    anchor = region['end'] + 1
                    bed_item['strand'] = '+'
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

def rename_chrom(bed,renamed_table):
    belong_table = {}
    for item in renamed_table.to_dict('record'):
        id_ = item['old_id']
        belong_table[id_] = item['new_id']

    redefined_bed = []
    for item in bed.to_dict('record'):
        bed_item = dict(item)
        bed_item['chr'] = belong_table[item['chr']]
        redefined_bed += [bed_item]

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    return redefined_bed

def main(bed_path,table_path,simple_mode,use_strand,flip,output_path):
    bed = read_bed(bed_path)
    renamed_table = pd.read_csv(table_path,sep='\t',dtype={'chr':str,'start':int,'end':int})
    if simple_mode:
        redefined_bed = rename_chrom(bed,renamed_table)
    else:    
        redefined_bed = redefine_coordinate(bed,renamed_table,use_strand,flip)
    write_bed(redefined_bed,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will redefine coordinate based on region file")
    parser.add_argument("-i", "--bed_path",help="Bed file to be redefined coordinate",required=True)
    parser.add_argument("-t", "--table_path",help="Table about renamed region",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined bed file",required=True)
    parser.add_argument("--simple_mode",action='store_true',
                       help="Just rename chromosome id, otherwise redefine coorindate data and rename chromosome id.")
    parser.add_argument("-s", "--use_strand",action='store_true')
    parser.add_argument("-f", "--flip",action="store_true",
                        help="Flip sequence info which are minus strand to plus strand")
                        
    
    args = parser.parse_args()
    main(args.bed_path,args.table_path,args.simple_mode,args.use_strand,args.flip,args.output_path)
    