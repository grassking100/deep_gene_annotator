import sys
import os
import pandas as pd
import numpy as np
import warnings
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser
from sequence_annotation.utils.utils import get_gff_item_with_attribute, read_bed, write_bed
from sequence_annotation.preprocess.bed2gff import bed_item2gff_item

def get_CDS_item(CDSs):
    #input one based data, return one based data
    cds_starts = [cds['start'] for cds in CDSs]
    cds_ends = [cds['end'] for cds in CDSs]
    min_start = min(cds_starts)
    max_end = max(cds_ends)
    indice = np.argsort(cds_starts)
    cds_starts = [cds_starts[index] for index in indice]
    cds_ends = [cds_ends[index] for index in indice]
    cds_sizes = [str(end-start+1) for start,end in zip(cds_starts,cds_ends)]
    cds_rel_starts = [str(start-min_start) for start in cds_starts]
    info = dict(CDSs[0])
    id_ = CDSs[0]['parent']
    #if id_ is not None:        
    info['id'] = id_
    info['rgb'] = '.'
    info['count'] = len(cds_sizes)
    info['block_size'] = ",".join(cds_sizes)
    info['block_related_start'] = ",".join(cds_rel_starts)
    info['thick_start'] = min_start
    info['thick_end'] = max_end
    return info
    
def get_CDS_bed(bed):
    bed_info_list = []
    error_ids = []
    for bed_item in bed:
        gff_items = bed_item2gff_item(bed_item)
        gff_items = [get_gff_item_with_attribute(gff_item) for gff_item in gff_items]
        CDSs = [item for item in gff_items if item['feature']=='CDS']
        if len(CDSs)>0:
            bed_info = get_CDS_item(CDSs)
            bed_info_list.append(bed_info)
        else:
            error_ids.append(bed_item['id'])
            #warnings.warn("{} will be discarded beacuse it lacks CDS region".format(bed_item['id']))
    bed = pd.DataFrame.from_dict(bed_info_list)
    return bed,error_ids
    
if __name__ =='__main__':
    parser = ArgumentParser(description="This program will get CDS in bed format")
    parser.add_argument("-i", "--input_path", help="Path of input bed file",required=True)
    parser.add_argument("-o", "--output_path", help="Path of output bed file",required=True)
    parser.add_argument("-e", "--error_id_path", help="Path to write discarded id")

    args = parser.parse_args()
    bed = BedInfoParser().parse(args.input_path)
    CDS_bed,error_ids = get_CDS_bed(bed)
    write_bed(CDS_bed,args.output_path)
    if args.error_id_path is not None:
        with open(args.error_id_path,'w') as fp:
            for id_ in error_ids:
                fp.write(id_)
