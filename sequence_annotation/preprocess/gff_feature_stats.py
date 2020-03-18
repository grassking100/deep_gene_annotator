import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.preprocess.utils import get_gff_with_intergenic_region,get_gff_with_intron,INTRON_TYPES
from sequence_annotation.utils.utils import get_gff_with_attribute, read_gff, read_fai,get_gff_with_feature_coord

def get_feature_stats(gff,fai):
    gff = gff[~gff['feature'].isin(INTRON_TYPES)]
    print("Get feature")
    gff = get_gff_with_attribute(gff)
    print("Get intron")
    gff = get_gff_with_intron(gff)
    print("Get intergenic region")
    gff = get_gff_with_intergenic_region(gff,fai)
    print("Get feature and coordinate")
    gff = get_gff_with_feature_coord(gff)
    print("Drop dupliacted")
    gff = gff.drop_duplicates('feature_coord')
    print("Start calculate")
    feature_group = gff.groupby('feature')
    stats = []
    length_dict = {}
    for feature,group in feature_group:
        lengths = list(group['end']-group['start']+1)
        stats_item = {'feature':feature}
        stats_item['number'] = len(group)
        stats_item['max'] = max(lengths)
        stats_item['min'] = min(lengths)
        stats_item['mean'] = np.mean(lengths)
        stats_item['std'] = np.std(lengths)
        stats_item['median'] = np.median(lengths)
        length_dict[feature] = lengths
        stats.append(stats_item)
    return stats,length_dict
        
def plot_length_stats(length_dict,saved_root):
    for feature,lengths in length_dict.items():
        plt.figure()
        plt.hist(lengths,100)
        plt.title("The distribution of {}'s length".format(feature))
        plt.xlabel("length")
        plt.ylabel("number")
        plt.savefig(os.path.join(saved_root,'{}_length.png'.format(feature)))
        
        plt.figure()
        plt.hist(np.log10(lengths),100)
        plt.title("The log distribution of {}'s length ".format(feature))
        plt.xlabel("log10(length)")
        plt.ylabel("number")
        plt.savefig(os.path.join(saved_root,'{}_log10_length.png'.format(feature)))
    
def main(gff_path,fai_path,saved_root):
    gff = read_gff(gff_path)
    fai = read_fai(fai_path)
    stats,lengths = get_feature_stats(gff,fai)
    stats_df = pd.DataFrame.from_dict(stats)
    stats_df.to_csv(os.path.join(saved_root,'feature_stats.tsv'),sep='\t',index=None)
    plot_length_stats(lengths,saved_root)
    
if __name__ =='__main__':
    parser = ArgumentParser(description="This program will convert gff file to bed file. "+
                                        "It will treat alt_acceptor and alt_donor regions as intron")
    parser.add_argument("-i", "--gff_path", help="Path of input gff file",required=True)
    parser.add_argument("-f", "--fai_path", help="Path of fai",required=True)
    parser.add_argument("-o", "--saved_root", help="Path of output file",required=True)
    args = parser.parse_args()
    main(args.gff_path,args.fai_path,args.saved_root)