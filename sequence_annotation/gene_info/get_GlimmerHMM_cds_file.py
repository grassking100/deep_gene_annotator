import sys
import os
import pandas as pd
import numpy as np
import warnings
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser
from sequence_annotation.utils.utils import get_gff_item_with_attribute, read_bed
from sequence_annotation.gene_info.bed2gff import bed_item2gff_item

def get_cds_list(CDSs):
    cds_starts = [cds['start'] for cds in CDSs]
    cds_ends = [cds['end'] for cds in CDSs]
    indice = np.argsort(cds_starts)
    cds_starts = [cds_starts[index] for index in indice]
    cds_ends = [cds_ends[index] for index in indice]
    info = dict(CDSs[0])
    chrom = info['chr']
    strand = info['strand']
    if strand not in ['+','-']:
        raise Exception("Unexcepted strand type: {}".format(strand))
    if strand=='-':
        cds_starts = reversed(cds_starts)
        cds_ends = reversed(cds_ends)
    cds_list = []
    for start, end in zip(cds_starts,cds_ends):
        if strand=='+':
            lhs,rhs = start ,end
        else:
            rhs,lhs = start ,end
        item = {'lhs':lhs,'rhs':rhs,'chr':chrom}
        cds_list.append(item)
    return cds_list
    
def get_cds_data(bed):
    cds_info = []
    for bed_item in bed:
        gff_items = bed_item2gff_item(bed_item)
        gff_items = [get_gff_item_with_attribute(gff_item) for gff_item in gff_items]
        CDSs = [item for item in gff_items if item['feature']=='CDS']
        if len(CDSs)>0:
            cds_list = get_cds_list(CDSs)
            cds_info.append(cds_list)
        else:
            warnings.warn("{} will be discarded beacuse it lacks CDS region".format(bed_item['id']))
    return cds_info
    
if __name__ =='__main__':
    parser = ArgumentParser(description="This program will get GlimmerHMM CDS file")
    parser.add_argument("-i", "--input_path", help="Path of input bed file",required=True)
    parser.add_argument("-o", "--output_path", help="Path of output GlimmerHMM CDS file",required=True)
    args = parser.parse_args()
    bed = BedInfoParser().parse(args.input_path)
    cds_info = get_cds_data(bed)
    with open(args.output_path,"w") as fp:
        for cds_list in cds_info:
            for item in cds_list:
                fp.write("{} {} {}\n".format(item['chr'],item['lhs'],item['rhs']))
            fp.write("\n")
