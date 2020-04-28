import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed
from sequence_annotation.preprocess.utils import read_region_table

def redefine_coordinate(bed,regions,chrom_target=None):
    bed = bed.copy()
    regions = regions.copy()
    chrom_target = chrom_target or 'ordinal_id_with_strand'
    redefined_bed = []
    bed['coord'] = bed['chr']+"_"+bed['strand']
    regions['coord'] = regions['chr']+"_"+regions['strand']
    region_dict = regions.groupby('coord')
    for bed_item in bed.to_dict('record'):
        regions = region_dict.get_group(bed_item['coord'])
        region = regions[(regions['start'] <= bed_item['start']) & 
                         (regions['end'] >= bed_item['end'])]
        if len(region) != 1:
            raise Exception("Cannot locate for region, {}".format(bed_item['coord']))
        region = region.iloc[0]
        anchor = region['start'] - 1
        bed_item['start'] -= anchor
        bed_item['end'] -= anchor
        bed_item['thick_start'] -= anchor
        bed_item['thick_end'] -= anchor
        bed_item['chr'] = region[chrom_target]
        redefined_bed.append(bed_item)

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    return redefined_bed

def main(bed_path,region_path,output_path,chrom_target=None):
    bed = read_bed(bed_path)
    regions = read_region_table(region_path)
    redefined_bed = redefine_coordinate(bed,regions,chrom_target)
    write_bed(redefined_bed,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will redefine coordinate data based on region file")
    parser.add_argument("-i", "--bed_path",help="Bed file to be redefined coordinate",required=True)
    parser.add_argument("-t", "--region_path",help="Table about renamed region",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined bed file",required=True)
    parser.add_argument("--chrom_target")

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)

    