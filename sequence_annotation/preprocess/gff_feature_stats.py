import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.preprocess.utils import get_gff_with_intergenic_region,get_gff_with_intron,INTRON_TYPES,read_region_table
from sequence_annotation.utils.utils import create_folder, get_gff_with_attribute, get_gff_with_feature_coord, read_gff

def get_feature_stats(gff,region_table,chrom_source):
    gff = gff[~gff['feature'].isin(INTRON_TYPES)]
    print("Get feature")
    gff = get_gff_with_attribute(gff)
    print("Get intron")
    gff = get_gff_with_intron(gff)
    print("Get intergenic region")
    gff = get_gff_with_intergenic_region(gff,region_table,chrom_source)
    print("Get feature and coordinate")
    gff = get_gff_with_feature_coord(gff)
    print("Drop duplicated")
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
        
def plot_length(length_dict,saved_root):
    for feature,lengths in length_dict.items():
        plt.figure()
        plt.hist(lengths,100)
        plt.title("The distribution of {}'s lengths".format(feature))
        plt.xlabel("length")
        plt.ylabel("number")
        plt.savefig(os.path.join(saved_root,'{}_length.png'.format(feature)))
        
def plot_log_length(length_dict,saved_root):
    for feature,lengths in length_dict.items():
        plt.figure()
        plt.hist(np.log10(lengths),100)
        plt.title("The log distribution of {}'s lengths".format(feature))
        plt.xlabel("log10(length)")
        plt.ylabel("number")
        plt.savefig(os.path.join(saved_root,'{}_log10_length.png'.format(feature)))
    
def main(gff_path,region_table_path,saved_root,chrom_source):
    create_folder(saved_root)
    gff = read_gff(gff_path)
    region_table = read_region_table(region_table_path)
    stats,lengths = get_feature_stats(gff,region_table,chrom_source)
    stats_df = pd.DataFrame.from_dict(stats)
    stats_df.to_csv(os.path.join(saved_root,'feature_stats.tsv'),sep='\t',index=None)
    plot_length(lengths,saved_root)
    plot_log_length(lengths,saved_root)
    
if __name__ =='__main__':
    parser = ArgumentParser(description="This program will write statistic data about GFF file")
    parser.add_argument("-i", "--gff_path", help="Path of input gff file",required=True)
    parser.add_argument("-r", "--region_table_path", help="Path of region table",required=True)
    parser.add_argument("-o", "--saved_root", help="Path of output file",required=True)
    parser.add_argument("-c","--chrom_source", type=str,required=True)
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
