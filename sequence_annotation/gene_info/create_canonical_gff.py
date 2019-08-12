import os, sys
import pandas as pd
import deepdish as dd
import warnings
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import get_gff_with_attribute, read_gff, write_gff
from sequence_annotation.gene_info.gff2bed import gff_info2bed_info,extract_orf
from sequence_annotation.gene_info.bed2gff import bed_item2gff_item
from sequence_annotation.gene_info.utils import get_id_table

def create_canonical_gff(gff,orf_region):
    gff = get_gff_with_attribute(gff)
    genes = gff[gff['feature']=='gene']
    ids = set(genes['id'])
    genes = genes.groupby('id')
    exons = gff[gff['feature'].isin(['exon','alt_alt_accept','alt_alt_donor'])]
    exons = exons.groupby('parent')
    gff_info_list = []
    for id_ in ids:
        gene = genes.get_group(id_).to_dict('record')[0]
        exons_ = exons.get_group(id_).to_dict('list')
        orf = {'id':id_,'thick_start':orf_region[id_]['start'],
               'thick_end':orf_region[id_]['end']}
        bed_info = gff_info2bed_info(gene,exons_,orf)
        #Convert one based to zero based
        bed_info['start'] -= 1
        bed_info['end'] -= 1
        bed_info['thick_start'] -= 1
        bed_info['thick_end'] -= 1
        bed_info['block_related_start'] = [int(site) for site in bed_info['block_related_start'].split(',')]
        bed_info['block_size'] = [int(size) for size in bed_info['block_size'].split(',')]
        gff_info = bed_item2gff_item(bed_info)
        length = 0
        for item in gff_info:
            if item['feature'] == 'CDS':
                length += (item['end'] - item['start'] + 1)
        if length%3 != 0:
            warnings.warn("{} will be discarded due to its CDS length".format(id_))
        else:    
            gff_info_list += gff_info

    gff = pd.DataFrame.from_dict(gff_info_list)
    return gff

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--alt_gff_path",required=True)
    parser.add_argument("-r", "--orf_region_path",required=True)
    parser.add_argument("-o", "--canonical_gff",required=True)
    parser.add_argument("-t", "--id_convert_path",required=False)
    
    args = parser.parse_args()
    orf_region = pd.read_csv(args.orf_region_path,sep='\t').to_dict('record')
    id_convert = None
    if args.id_convert_path is not None:
        id_convert = get_id_table(args.id_convert_path)
    gene_orf_region = {}
    for item in orf_region:
        gene_id = item['id']
        if id_convert is not None:
            gene_id = id_convert[gene_id]
        gene_orf_region[gene_id] = {'start':item['start'],'end':item['end']}

    gff = read_gff(args.alt_gff_path)
    gff = create_canonical_gff(gff,gene_orf_region)
    write_gff(gff,args.canonical_gff)