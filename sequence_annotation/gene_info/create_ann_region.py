import os
import sys
import deepdish as dd
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.genome_handler.create_ann_region import create_ann_region
from sequence_annotation.genome_handler.alt_anns_creator import Gff2AnnSeqs
from sequence_annotation.genome_handler.mediator import Mediator
from sequence_annotation.utils.utils import read_fai, read_gff, get_gff_with_seq_id
from sequence_annotation.genome_handler.utils import ann_count

def create_ann_region(alt_region_gff_path,fai_path,source_name):
    #Read chromosome length file
    genome_info = read_fai(fai_path)
    #Parse the alt region file and convert its data to AnnSeqContainer
    gff = read_gff(alt_region_gff_path)
    gff = get_gff_with_seq_id(gff)
    converter = Gff2AnnSeqs()
    genome = converter.convert(gff,genome_info,source_name)
    print(alt_region_gff_path)
    print(ann_count(genome))
    return genome

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will used alt region gff data to create annotated genome\n"+
                            "and it will selecte region from annotated genome according to the\n"+
                            "selected region and the result will be saved to output_path in h5 format")
    parser.add_argument("-i", "--alt_region_gff_path",
                        help="Path of selected mRNA gff file",required=True)
    parser.add_argument("-f", "--fai_path",
                        help="Path of fai file",required=True)
    parser.add_argument("-s", "--source_name",
                        help="Genome source name",required=True)
    parser.add_argument("-o", "--output_path",
                        help="Path of output file",required=True)
    args = parser.parse_args()
    parsed = create_ann_region(args.alt_region_gff_path,args.fai_path,args.source_name)
    dd.io.save(args.output_path,parsed.to_dict())
