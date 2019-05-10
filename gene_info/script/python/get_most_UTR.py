import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import get_bed_most_UTR,read_bed,write_bed
import pandas as pd
from argparse import ArgumentParser
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-b","--bed_path",
                        help="bed_path",required=True)
    parser.add_argument("-s", "--saved_root",
                        help="saved_root",required=True)
    args = vars(parser.parse_args())
    bed_path = args['bed_path']
    saved_root = args['saved_root']
    if os.path.exists(saved_root+"/most_five_UTR.tsv") and os.path.exists(saved_root+"/most_three_UTR.tsv"):
        print("Result files are already exist, procedure will be skipped.")
    else:
        bed = read_bed(bed_path)
        utr_bed = get_bed_most_UTR(bed)
        utr_bed['chr'] = utr_bed['chr'].str.replace('Chr','')
        utr_bed['score']='.'
        five_most_utr = utr_bed[utr_bed['type']=='five_most_utr']
        three_most_utr = utr_bed[utr_bed['type']=='three_most_utr']
        five_most_utr.to_csv(saved_root+"/most_five_UTR.tsv",sep='\t',index=None)
        three_most_utr.to_csv(saved_root+"/most_three_UTR.tsv",sep='\t',index=None)
        five_most_utr_bed = five_most_utr[['chr','start','end','id','score','strand']].copy()
        three_most_utr_bed = three_most_utr[['chr','start','end','id','score','strand']].copy()
        write_bed(five_most_utr_bed,saved_root+"/most_five_UTR.bed")
        write_bed(three_most_utr_bed,saved_root+"/most_three_UTR.bed")
