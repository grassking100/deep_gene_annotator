import os
import sys
import deepdish as dd
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.genome_handler.create_ann_region import create_ann_region
from argparse import ArgumentParser

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will used mRNA_bed12 data to create annotated genome\n"+
                            "and it will selecte region from annotated genome according to the\n"+
                            "selected_region and selected region will save to output_path in h5 format")
    parser.add_argument("-i", "--mRNA_bed12_path",
                        help="Path of selected mRNA bed file",required=True)
    parser.add_argument("-f", "--fai_path",
                        help="Path of fai file",required=True)
    parser.add_argument("-s", "--source_name",
                        help="Genome source name",required=True)
    parser.add_argument("-o", "--output_path",
                        help="Path of output file",required=True)
    args = parser.parse_args()
    parsed = create_ann_region(args.mRNA_bed12_path,args.fai_path,args.source_name)
    dd.io.save(args.output_path,parsed.to_dict())
