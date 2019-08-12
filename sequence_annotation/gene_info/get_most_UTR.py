import os, sys
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from utils import get_bed_most_UTR
from sequence_annotation.utils.utils import read_bed,write_bed

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-b","--bed_path",
                        help="bed_path",required=True)
    parser.add_argument("-s", "--saved_root",
                        help="saved_root",required=True)
    args = parser.parse_args()

    five_UTR_tsv_path = os.path.join(args.saved_root,"most_five_UTR.tsv")
    three_UTR_tsv_path = os.path.join(args.saved_root,"most_three_UTR.tsv")
    five_UTR_bed_path = os.path.join(args.saved_root,"most_five_UTR.bed")
    three_UTR_bed_path = os.path.join(args.saved_root,"most_three_UTR.bed")

    if os.path.exists(five_UTR_bed_path) and os.path.exists(three_UTR_bed_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        bed = read_bed(args.bed_path)
        utr_bed = get_bed_most_UTR(bed)
        utr_bed['chr'] = utr_bed['chr'].str.replace('Chr','')
        utr_bed['score']='.'
        five_most_utr = utr_bed[utr_bed['type']=='five_most_utr']
        three_most_utr = utr_bed[utr_bed['type']=='three_most_utr']
        five_most_utr.to_csv(five_UTR_tsv_path,sep='\t',index=None)
        three_most_utr.to_csv(three_UTR_tsv_path,sep='\t',index=None)
        five_most_utr_bed = five_most_utr[['chr','start','end','id','score','strand']].copy()
        three_most_utr_bed = three_most_utr[['chr','start','end','id','score','strand']].copy()
        write_bed(five_most_utr_bed,five_UTR_bed_path)
        write_bed(three_most_utr_bed,three_UTR_bed_path)
