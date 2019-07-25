import pandas as pd
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_gff
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser

def bed_item2gff_item(item,id_convert):
    gff_items = []
    thick_start = item['thick_start'] + 1
    thick_end = item['thick_end']
    mRNA_id = item['id']
    gene_id = id_convert[id_]
    basic_block = {'chr':item['chr'],'strand':item['strand'],'source':item['source'],
                   'score':item['score'],'frame':'.'}
    gene = dict(basic_block)
    gene['start'], gene['end'] = item['start'] + 1, item['end']
    gene['feature'] = 'gene'
    gene['attribute'] = 'ID={}'.format(gene_id)
    mRNA = dict(gene)
    mRNA['feature'] = 'mRNA'
    mRNA['attribute'] = "ID={};Parent={}".format(mRNA_id,gene_id)
    gff_items.append(gene)
    gff_items.append(mRNA)
    
    for index in range(item['count']):
        abs_start = item['block_related_start'][index] + gene['start']
        exon = dict(basic_block)
        exon['start'] = abs_start
        exon['end'] = abs_start+item['block_size'][index]-1
        exon['feature'] = 'exon'
        exon['attribute'] = 'Parent={}'.format(mRNA_id)
        gff_items.append(exon)
        #If Coding region is exist
        if thick_start<=thick_end:
            cds_site = []
            if exon['start'] <= thick_start <= exon['end']:
                cds_site.append(thick_start)
            if exon['start'] <= thick_end <= exon['end']:
                cds_site.append(thick_end)
            if len(cds_site)==0:
                if thick_start <= exon['start'] and exon['end'] <= thick_end:
                    cds = dict(exon)    
                    cds['feature'] = 'CDS'
                    gff_items.append(cds)
                else:    
                    utr = dict(exon)
                    utr['feature'] = 'UTR'
                    gff_items.append(utr)
            elif len(cds_site)==2:
                utr = dict(exon)
                utr['feature'] = 'UTR'
                utr_1 = dict(utr)
                utr_2 = dict(utr)
                utr_1['end'], utr_2['start'] = thick_start - 1, thick_end + 1
                cds = dict(exon)
                cds['feature'] = 'CDS'
                cds['start'],cds['end'] = thick_start, thick_end
                gff_items.append(cds)
                gff_items.append(utr_1)
                gff_items.append(utr_2)
            else:
                if exon['start']<= thick_start <= exon['end']:
                    utr = dict(exon)
                    utr['feature'] = 'UTR'
                    utr = dict(utr)
                    utr['end'] = thick_start - 1
                    cds = dict(exon)
                    cds['feature'] = 'CDS'
                    cds['start'],cds['end'] = thick_start , exon['end']
                    gff_items.append(cds)
                    gff_items.append(utr)
                else:
                    utr = dict(exon)
                    utr['feature'] = 'UTR'
                    utr = dict(utr)
                    utr['start'] = thick_end + 1
                    cds = dict(exon)
                    cds['feature'] = 'CDS'
                    cds['start'],cds['end'] = exon['start'], thick_end
                    gff_items.append(cds)
                    gff_items.append(utr)
    return gff_items
    
def bed2gff(bed,id_convert):
    gff_items = {}
    basic_items = []
    for item in bed:
        basic_items.append(bed_item2gff_item(item,id_convert))
    for basic_item in basic_items:
        gene_id = basic_item[0]['id']
        if gene_id not in gff_items.keys():
            gff_items[gene_id] = [basic_item[0]]
        gff_items[gene_id] += basic_item[1:]
    gff_list = []
    for list_ in gff_items.values():
        gff_list += list_
    gff = pd.DataFrame.from_dict(gff_list)
    return gff

if __name__ =='__main__':
    bed_path=sys.argv[1]
    gff_path=sys.argv[2]
    id_path=sys.argv[3]
    parser = BedInfoParser()
    bed = parser.parse(bed_path)
    id_convert = pd.read_csv(id_path,sep='\t',index_col=0).to_dict()['gene_id']
    gff = bed2gff(bed,id_convert)
    write_gff(gff,gff_path)
