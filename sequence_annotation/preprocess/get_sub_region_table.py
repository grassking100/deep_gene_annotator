import os,sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.preprocess.utils import read_region_table

def main(region_table_path,chrom_id_path,chrom_source,output_path):
    region_table = read_region_table(region_table_path)
    chrom_ids = pd.read_csv(chrom_id_path,header=None)
    chrom_ids = list(chrom_ids[0])
    #print(chrom_source,region_table.head)
    subtable = region_table[region_table[chrom_source].isin(chrom_ids)]
    subtable.to_csv(output_path,index=None,sep='\t')

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-r", "--region_table_path",required=True)
    parser.add_argument("-i", "--chrom_id_path",required=True)
    parser.add_argument("-s", "--chrom_source",required=True)
    parser.add_argument("-o", "--output_path",required=True)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
