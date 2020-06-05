import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder, read_bed, read_gff
from sequence_annotation.utils.stats import get_stats
from sequence_annotation.preprocess.utils import RNA_TYPES
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict
from sequence_annotation.preprocess.bed2gff import bed2gff
from sequence_annotation.preprocess.site_analysis import get_site_diff

def plot_hist(title, path, values):
    plt.figure()
    plt.hist(values, 100)
    plt.title(title)
    plt.xlabel("distance")
    plt.ylabel("number")
    plt.savefig(path)

def main(bed_path,id_table_path,tss_path,cs_path,output_root):
    ###Read file###
    create_folder(output_root)
    id_dict = get_id_convert_dict(id_table_path)
    reference_bed = read_bed(bed_path)
    tss_sites = read_gff(tss_path)
    cleavage_sites = read_gff(cs_path)
    ##
    reference_gff = bed2gff(reference_bed,id_dict)
    rna_gff = reference_gff[reference_gff['feature'].isin(RNA_TYPES)]
    tss_site_e_to_r_diff = get_site_diff(rna_gff,tss_sites,multiprocess=40)
    tss_site_r_to_e_diff = get_site_diff(rna_gff,tss_sites,answer_as_ref=False,multiprocess=40)
    cleavage_site_e_to_r_diff = get_site_diff(rna_gff,cleavage_sites,five_end=False,multiprocess=40)
    cleavage_site_r_to_e_diff = get_site_diff(rna_gff,cleavage_sites,five_end=False,
                                              answer_as_ref=False,multiprocess=40)
    stats_result = {}
    stats_result['tss_site_r_to_e_diff'] = get_stats(np.abs(tss_site_r_to_e_diff))
    stats_result['tss_site_e_to_r_diff'] = get_stats(np.abs(tss_site_e_to_r_diff))
    stats_result['cleavage_site_r_to_e_diff'] = get_stats(np.abs(cleavage_site_r_to_e_diff))
    stats_result['cleavage_site_e_to_r_diff'] = get_stats(np.abs(cleavage_site_e_to_r_diff))
    
    siter_diff_path = os.path.join(output_root, "site_abs_diff.tsv")
    pd.DataFrame.from_dict(stats_result).T.to_csv(siter_diff_path,index_label='stats',sep='\t')

    tss_title='The distance from the closest reference TSS to experimental TSS'
    tss_path=os.path.join(output_root, 'tss_site_r_to_e_diff.png')
    plot_hist(tss_title,tss_path,tss_site_r_to_e_diff)
    
    tss_ref_title='The distance from the closest experimental TSS to reference TSS'
    tss_ref_path=os.path.join(output_root, 'tss_site_e_to_r_diff.png')
    plot_hist(tss_ref_title,tss_ref_path,tss_site_e_to_r_diff)

    cs_title='The distance from the closest reference CS to experimental CS'
    cs_path=os.path.join(output_root, 'cleavage_site_r_to_e_diff.png')
    plot_hist(cs_title,cs_path,cleavage_site_r_to_e_diff)

    cs_ref_title='The distance from the closest experimental CS to reference CS'
    cs_ref_path=os.path.join(output_root, 'cleavage_site_e_to_r_diff.png')
    plot_hist(cs_ref_title,cs_ref_path,cleavage_site_e_to_r_diff)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-b","--bed_path",required=True,
                        help="Path of selected official gene info file")
    parser.add_argument("-t","--tss_path",required=True,
                        help="Path of TSS file")
    parser.add_argument("-c","--cs_path",required=True,
                        help="Path of CS file")
    parser.add_argument("-o","--output_root",required=True,
                        help="Path to save")
    parser.add_argument("-i","--id_table_path",required=True)
    args = parser.parse_args()
    kwargs=vars(args)
    main(**kwargs)
