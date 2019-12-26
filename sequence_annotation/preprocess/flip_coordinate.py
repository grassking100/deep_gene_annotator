import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed,read_fai,write_gff,read_gff

def flip_bed(bed,regions,fai):
    redefined_bed = []
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

def flip_gff(gff,regions,fai):
    redefined_gff = []
    for item in gff.to_dict('record'):
        gff_item = dict(item)
        region = regions[regions['old_id'] == gff_item['chr']]
        if len(region) != 1:
            raise Exception("Cannot locate for {}".format(gff_item['chr']))
        region = region.to_dict('record')[0]
        if region['strand'] == '-':
            gff_item['strand'] = '-'
            anchor = fai[gff_item['chr']] + 1
            #Start recoordinate
            gff_item['end'] = anchor - item['start']
            gff_item['start'] = anchor - item['end']
        gff_item['chr'] = region['new_id']
        redefined_gff.append(gff_item)

    redefined_gff = pd.DataFrame.from_dict(redefined_gff)
    return redefined_gff

def main(input_path,region_path,output_path,fai_path):
    fai = read_fai(fai_path)
    regions = pd.read_csv(region_path,sep='\t',dtype={'chr':str,'start':int,'end':int})
    if 'bed' in input_path.split('.')[-1]:
        bed = read_bed(input_path)
        redefined_bed = flip_bed(bed,regions,fai)
        write_bed(redefined_bed,output_path)
    else:
        gff = read_gff(input_path)
        redefined_gff = flip_gff(gff,regions,fai)
        write_gff(redefined_gff,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will would used table information"
                            "to check regions' original strand status, its strand would "
                            "be set to original strand and it's id would be renamed, "
                            "the coordinate data on plus strand but originally on minus "
                            "strand would be flipped to minus strand")
    parser.add_argument("-i", "--input_path",help="File to be redefined coordinate",required=True)
    parser.add_argument("-t", "--region_path",help="Table about renamed region",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined file",required=True)
    parser.add_argument("--fai_path",help="Path of fai file",required=True)
    
    args = parser.parse_args()
    main(args.input_path,args.region_path,args.output_path,args.fai_path)
    