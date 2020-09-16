import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.file_process.utils import read_gff,write_gff
from sequence_annotation.file_process.get_region_table import read_region_table

def redefine_coordinate(gff,region_table,with_strand=False):
    gff = gff.copy()
    region_table = region_table.copy()

    if with_strand:
        chrom_target = 'ordinal_id_with_strand'
    else:
        chrom_target = 'ordinal_id_wo_strand'
    redefined_gff = []
    gff['coord'] = gff['chr']+"_"+gff['strand']
    region_table['coord'] = region_table['chr']+"_"+region_table['strand']
    gff_groups = gff.groupby('coord')
    for index,region in region_table.iterrows():
        gff_group = gff_groups.get_group(region['coord'])
        gff_group = gff_group[(region['start'] <= gff_group['start']) & 
                              (region['end'] >= gff_group['end'])]
        gff_group = gff_group.copy()
        gff_group['chr'] = region[chrom_target]
        if with_strand:
            if region['strand'] == '+':
                anchor = region['start']
                gff_group['start'] -= anchor
                gff_group['end'] -= anchor
            else:
                anchor = region['end']
                start = gff_group['start'].copy()
                end = gff_group['end'].copy()
                gff_group['end'] = anchor - start
                gff_group['start'] = anchor - end
                gff_group['strand'] = '+'
        else:
            anchor = region['start']
            gff_group['start'] -= anchor
            gff_group['end'] -= anchor
        redefined_gff.append(gff_group)
    redefined_gff = pd.concat(redefined_gff)
    redefined_gff['start'] += 1
    redefined_gff['end'] += 1
    del redefined_gff['coord']
    return redefined_gff

def main(input_path,region_table_path,output_path,chrom_target=None):
    gff = read_gff(input_path)
    region_table = read_region_table(region_table_path)
    redefined_gff = redefine_coordinate(gff,region_table,chrom_target)
    write_gff(redefined_gff,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will redefine coordinate data based on region file")
    parser.add_argument("-i", "--input_path",help="GFF file to be redefined coordinate",required=True)
    parser.add_argument("-t", "--region_table_path",help="Table about renamed region",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined GFF file",required=True)
    parser.add_argument("--with_strand",action='store_true')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)

    