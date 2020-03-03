import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../..")
import numpy as np
import pandas as pd
from argparse import ArgumentParser
    
def write_bed_length_stats(bed_path,output_path):
    bed = pd.read_csv(bed_path,header=None,sep='\t')
    lengths = list(bed[2] - bed[1])

    with open(output_path,"w") as fp:
        fp.write("Max length: {}\n".format(max(lengths)))
        fp.write("Min length: {}\n".format(min(lengths)))
        fp.write("Median length: {}\n".format(np.median(lengths)))
        fp.write("Mean length: {}\n".format(np.mean(lengths)))

if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--bed_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    args = parser.parse_args()
    
    write_bed_length_stats(args.bed_path,args.output_path)
