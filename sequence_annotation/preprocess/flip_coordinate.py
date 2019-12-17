import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed,read_fai

def flip_coordinate(bed,regions,fai):
    origin_strand = {}
    rename_table = {}
    redefined_bed = []
    for item in regions.to_dict('record'):
        id_ = item['old_id']
        origin_strand[id_] = item['strand']
        rename_table[id_] = item['new_id']
    for item in bed.to_dict('record'):
        bed_item = dict(item)
        region = regions[regions['old_id'] == bed_item['chr']]
        if len(region) != 1:
            raise Exception("Cannot locate for {}".format(bed_item['chr']))
        region = region.to_dict('record')[0]
        if region['strand'] == '-':
            bed_item['strand'] = '-'
            anchor = fai[bed_item['chr']] + 1
            block_starts = [int(v)+bed_item['start'] for v in  bed_item['block_related_start'].split(",")]
            block_sizes = [v for v in  bed_item['block_size'].split(",")]
            block_ends = [block_start + int(block_size) -1 for block_start,block_size in zip(block_starts,block_sizes)]
            #Start recoordinate
            bed_item['end'] = anchor - item['start']
            bed_item['start'] = anchor - item['end']
            bed_item['thick_end'] = anchor - item['thick_start']
            bed_item['thick_start'] = anchor - item['thick_end']
            new_block_starts = [str(anchor - block_end - bed_item['start']) for block_end in block_ends]
            bed_item['block_related_start'] = ','.join(reversed(new_block_starts))
            bed_item['block_size'] = ','.join(reversed(block_sizes))
        bed_item['chr'] = region['new_id']
        redefined_bed.append(bed_item)

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    return redefined_bed

def main(input_path,region_path,output_path,fai_path):
    bed = read_bed(input_path)
    fai = read_fai(fai_path)
    regions = pd.read_csv(region_path,sep='\t',dtype={'chr':str,'start':int,'end':int})
    redefined_bed = flip_coordinate(bed,regions,fai)
    write_bed(redefined_bed,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will would used table information"
                            "to check regions' original strand status, its strand would "
                            "be set to original strand and it's id would be renamed, "
                            "the coordinate data on plus strand but originally on minus "
                            "strand would be flipped to minus strand")
    parser.add_argument("-i", "--input_path",help="Bed file to be redefined coordinate",required=True)
    parser.add_argument("-t", "--region_path",help="Table about renamed region",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined bed file",required=True)
    parser.add_argument("--fai_path",help="Path of fai file",required=True)
    
    args = parser.parse_args()
    main(args.input_path,args.region_path,args.output_path,args.fai_path)
    