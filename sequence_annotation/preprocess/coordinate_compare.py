import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed

def main(compared_bed_path,comparing_bed_path,output_path):
    compared_bed = read_bed(compared_bed_path)
    comparing_bed = read_bed(comparing_bed_path)
    compared_bed_group = compared_bed.groupby('id')
    comparing_bed_group = comparing_bed.groupby('id')
    strands = set(list(compared_bed['strand'])+list(comparing_bed['strand']))
    if strands != set(['+','-']):
        raise Exception()

    result = []
    for id_,compared_item in compared_bed_group:
        if id_ not in comparing_bed_group.groups.keys():
            result.append({'id':id_,'tss_diff':None,'cleavage_site_diff':None})
        else:
            comparing_item = comparing_bed_group.get_group(id_)
            if len(compared_item)==1 and len(comparing_item)==1:
                strand = list(compared_item['strand'])[0]
                start_diff = list(comparing_item['start'])[0]-list(compared_item['start'])[0]
                end_diff = list(comparing_item['end'])[0]-list(compared_item['end'])[0]
                if strand == '+':
                    tss_diff = start_diff
                    cs_diff = end_diff
                else:
                    tss_diff = -end_diff
                    cs_diff = -start_diff
                result.append({'id':id_,'tss_diff':tss_diff,'cleavage_site_diff':cs_diff})
            else:
                raise Exception()
    result = pd.DataFrame.from_dict(result)[['id','tss_diff','cleavage_site_diff']]
    result.to_csv(output_path,sep='\t',index=None)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will compare tss cand cs between two files")
    parser.add_argument("-r", "--compared_bed_path",help="Bed file to compared",required=True)
    parser.add_argument("-c", "--comparing_bed_path",help="Bed file to comparing",required=True)
    parser.add_argument("-o", "--output_path",help="Path of compared result",required=True)

    args = parser.parse_args()
    main(args.compared_bed_path,args.comparing_bed_path,args.output_path)
    