import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed

def redefine_coordinate(bed,regions):
    redefined_bed = []
    for bed_item_ in bed.to_dict('record'):
        match_strand = regions['strand'] == bed_item_['strand']
        match_chr = regions['chr'] == bed_item_['chr']
        match_start = regions['start'] <= bed_item_['start']
        match_end = regions['end'] >= bed_item_['end']
        selected_regions = regions[(match_strand) & (match_chr) & (match_start) & (match_end)]
        if len(selected_regions) != 1:
            raise Exception("Cannot locate for region, {}".format(bed_item_['id']))
        region = selected_regions.to_dict('record')[0]
        bed_item = dict(bed_item_)
        anchor = region['start'] - 1
        bed_item['start'] -= anchor
        bed_item['end'] -= anchor
        bed_item['thick_start'] -= anchor
        bed_item['thick_end'] -= anchor
        bed_item['chr'] = region['new_id']
        redefined_bed.append(bed_item)

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    return redefined_bed

def main(bed_path,region_path,output_path):
    bed = read_bed(bed_path)
    regions = pd.read_csv(region_path,sep='\t',dtype={'chr':str,'start':int,'end':int})
    redefined_bed = redefine_coordinate(bed,regions)
    write_bed(redefined_bed,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will redefine coordinate data based on region file")
    parser.add_argument("-i", "--bed_path",help="Bed file to be redefined coordinate",required=True)
    parser.add_argument("-t", "--region_path",help="Table about renamed region",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined bed file",required=True)

    args = parser.parse_args()
    main(args.bed_path,args.region_path,args.output_path)
    