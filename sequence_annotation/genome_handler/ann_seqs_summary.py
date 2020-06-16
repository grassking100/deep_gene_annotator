import os, sys
import numpy as np
import deepdish as dd
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder,write_json
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.select_data import classify_ann_seqs
from sequence_annotation.genome_handler.ann_genome_processor import class_count

def plot_log10_length(length_dict,output_root):
    for feature,lengths in length_dict.items():
        plt.figure()
        plt.hist(np.log10(lengths),100)
        plt.title("The log distribution of length (nt) of\n{}".format(feature),fontsize=16)
        plt.xlabel("log10(length)",fontsize=16)
        plt.ylabel("number",fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt_path=os.path.join(output_root,'{}_log10_length.png'.format(feature.replace(' ','_')))
        plt.savefig(plt_path,bbox_inches = 'tight',pad_inches = 0)

def main(input_path, output_root):
    create_folder(output_root)
    ann_seqs = dd.io.load(input_path)
    ann_seqs = AnnSeqContainer().from_dict(ann_seqs)
    write_json(class_count(ann_seqs),os.path.join(output_root,'label_count.json'))
    data = classify_ann_seqs(ann_seqs)
    multiple_exon_region_anns = data['multiple_exon_region']
    single_exon_region_anns = data['single_exon_region']
    no_exon_region_anns = data['no_exon_region']
    lengths = {}
    lengths['all regions'] = [ann.length for ann in ann_seqs]
    lengths['regions with multiple exons'] = [ann.length for ann in multiple_exon_region_anns]
    lengths['regions with single exon'] = [ann.length for ann in single_exon_region_anns]
    lengths['regions with no exon'] = [ann.length for ann in no_exon_region_anns]
    
    plot_log10_length(lengths,output_root)
    stats = {'median':{},'count':{},'max':{},'min':{}}
    for type_,lengths_ in lengths.items():
        stats['count'][type_.replace(' ','_')] = int(len(lengths_))
        stats['median'][type_.replace(' ','_')] = int(np.median(lengths_))
        stats['max'][type_.replace(' ','_')] = int(max(lengths_))
        stats['min'][type_.replace(' ','_')] = int(min(lengths_))
    
    write_json(stats,os.path.join(output_root,'length_stats.json'))
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", help='The path of annotation in h5 format', required=True)
    parser.add_argument("-o", "--output_root", required=True)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
