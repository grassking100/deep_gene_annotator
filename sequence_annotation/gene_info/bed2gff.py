import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../..")
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import write_gff
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser

strand_convert = {'plus':'+','minus':'-','+':'+','-':'-'}

def bed_item2gff_item(item,id_convert=None):
    #input zero based data, return one based data
    gff_items = []
    thick_start = item['thick_start'] + 1
    thick_end = item['thick_end'] + 1
    mRNA_id = item['id']
    if id_convert is not None:
        gene_id = id_convert[mRNA_id]
    else:    
        gene_id = mRNA_id
    strand = strand_convert[item['strand']]
    basic_block = {'chr':item['chr'],'strand':strand,'source':'.',
                   'score':item['score'],'frame':'.'}
    gene = dict(basic_block)
    gene['start'], gene['end'] = item['start'] + 1, item['end'] + 1
    gene['feature'] = 'gene'
    gene['id'] = gene_id
    gene['attribute'] = 'ID={}'.format(gene_id)
    mRNA = dict(gene)
    mRNA['feature'] = 'mRNA'
    gene['id'] = mRNA_id
    mRNA['attribute'] = "ID={};Parent={}".format(mRNA_id,gene_id)
    gff_items.append(gene)
    gff_items.append(mRNA)
    exons = []
    for index in range(item['count']):
        abs_start = item['block_related_start'][index] + gene['start']
        exon = dict(basic_block)
        exon['start'] = abs_start
        exon['end'] = abs_start+item['block_size'][index]-1
        exon['feature'] = 'exon'
        exon['attribute'] = 'Parent={}'.format(mRNA_id)
        exons.append(exon)

    for index,exon in enumerate(exons):
        gff_items.append(exon)
        #Create intron info
        if index < len(exons)-1:
            next_exon = exons[index+1]
            intron = dict(exon)
            intron['start'] = exon['end'] + 1
            intron['end'] = next_exon['start'] - 1
            intron['feature'] = 'intron'
            gff_items.append(intron)
            
        #If Coding region is exist, create UTR or CDS info
        if thick_start<=thick_end:
            cds_site = []
            if exon['start'] <= thick_start <= exon['end']:
                cds_site.append(thick_start)
            if exon['start'] <= thick_end <= exon['end']:
                cds_site.append(thick_end)
            if len(cds_site)==0:
                whole_block = dict(exon)
                if thick_start <= exon['start'] and exon['end'] <= thick_end:
                    whole_block['feature'] = 'CDS'
                else:    
                    whole_block['feature'] = 'UTR'
                gff_items.append(whole_block)
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
                    utr['end'] = thick_start - 1
                    cds = dict(exon)
                    cds['feature'] = 'CDS'
                    cds['start'],cds['end'] = thick_start , exon['end']
                    gff_items.append(cds)
                    gff_items.append(utr)
                else:
                    utr = dict(exon)
                    utr['feature'] = 'UTR'
                    utr['start'] = thick_end + 1
                    cds = dict(exon)
                    cds['feature'] = 'CDS'
                    cds['start'],cds['end'] = exon['start'], thick_end
                    gff_items.append(cds)
                    gff_items.append(utr)
        #Create UTR info
        else:
            whole_block = dict(exon)  
            whole_block['feature'] = 'UTR'
            gff_items.append(whole_block)

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
    parser = ArgumentParser()
    parser.add_argument("-i", "--bed_path",required=True)
    parser.add_argument("-o", "--gff_output",required=True)
    parser.add_argument("-t", "--id_table_path",required=False)
    args = parser.parse_args()
    parser = BedInfoParser()
    bed = parser.parse(args.bed_path)
    if args.id_table_path is not None:
        id_convert = pd.read_csv(args.id_table_path,sep='\t',index_col=0).to_dict()['gene_id']
    else:
        id_convert = None
    gff = bed2gff(bed,id_convert)
    write_gff(gff,args.gff_output)
