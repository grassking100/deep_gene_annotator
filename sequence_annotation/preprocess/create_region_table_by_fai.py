import os,sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_fai

def get_region_table(fai):
    table = []
    for chrom_id,length in fai.items():
        for strand in ['+','-']:
            item = {'new_id':'.','old_id':'.',
                    'chr':chrom_id,'strand':strand,
                    'start':1,'end':length}
            table.append(item)    
    table = pd.DataFrame.from_dict(table)
    table = table[['new_id','old_id','chr','strand','start','end']]
    return table    

def main(fai_path,region_table_path):
    fai = read_fai(fai_path)
    region_table = get_region_table(fai)
    region_table.to_csv(region_table_path,index=None,sep='\t')

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will create region table by fai files")
    parser.add_argument("-i", "--fai_path",required=True)
    parser.add_argument("-o", "--region_table_path",required=True)
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
