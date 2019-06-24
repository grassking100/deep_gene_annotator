import pandas as pd
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))

def bed_item2gff_item(item,id_convert):
    gff_items = []
    chrom = item[0]
    start = item[1] + 1
    end = item[2]
    id_ = item[3]
    gene_id = id_convert[id_]
    source = item[4]
    strand = item[5]
    thick_start = item[6] + 1
    thick_end = item[7]
    score = item[8]
    block_num = item[9]
    block_size = [int(v) for v in item[10].split(',')]
    block_rel_start = [int(v) for v in item[11].split(',')]
    mRNA = {'chr':chrom,'start':start,'end':end,
            'strand':strand,'source':source,'frame':'.',
            'feature':'mRNA','score':score,
            'attribute':'ID='+id_+";Parent="+gene_id,'id':id_,'parent':'.'}
    gene = {'chr':chrom,'start':start,'end':end,
            'strand':strand,'source':source,'frame':'.',
            'feature':'gene','score':'.',
            'attribute':'ID='+gene_id,'id':gene_id,'parent':'.'}
    gff_items.append(mRNA)    
    for index in range(block_num):
        exon = {'chr':chrom,'start':start+block_rel_start[index],
                'end':start+block_rel_start[index]+block_size[index]-1,
                'strand':strand,'source':source,
                'feature':'exon','score':score,
                'attribute':'Parent='+id_,
                'frame':'.','id':'.','parent':id_}
        gff_items.append(exon)
    return gff_items
    
def bed2gff(bed_path,gff_path,id_convert):
    result = pd.read_csv(bed_path,sep='\t',header=None)
    gff_items = []
    for item in result.to_dict('record'):
        gff_items += bed_item2gff_item(item,id_convert)
        break
    mRNA_items = [item for item in gff_items if item['feature']=='mRNA']
    gene_items = {}
    for item in mRNA_items:
        temp = dict(item)
        id_ = id_convert[item['id']]
        if id_ not in gene_items.keys():
            temp['id'] = id_
            temp['parent'] = '.'
            temp['feature'] = 'gene'
            temp['attribute'] = 'ID='+id_
            gene_items[temp['id']] = temp
    gff_items += gene_items.values()
    gff = pd.DataFrame.from_dict(gff_items)[['chr','source','feature',
                                             'start','end','score',
                                             'strand','frame','attribute']]
    fp = open(gff_path, 'w')
    fp.write("##gff-version 3\n")
    gff.to_csv(fp,header=None,sep='\t',index=None)
    fp.close()
if __name__ =='__main__':
    bed_path=sys.argv[1]
    gff_path=sys.argv[2]
    id_path=sys.argv[3]
    id_convert = pd.read_csv(id_path,sep='\t',index_col=0).to_dict()['gene_id']
    bed2gff(bed_path,gff_path,id_convert)