import sys,os
import pandas as pd
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import BED_COLUMNS,GFF_COLUMNS
from sequence_annotation.utils.utils import get_gff_with_attribute, get_gff_with_feature_coord

preprocess_src_root = os.path.dirname(os.path.abspath(__file__))
    
GENE_TYPES = ['gene','transposable_element','transposable_element_gene','pseudogene']
RNA_TYPES = ['mRNA','pseudogenic_tRNA','pseudogenic_transcript','antisense_lncRNA','lnc_RNA',
             'antisense_RNA','transcript_region','transposon_fragment','miRNA_primary_transcript',
             'tRNA','snRNA','ncRNA','snoRNA','rRNA','transcript']
EXON_TYPES = ['exon','pseudogenic_exon']
SUBEXON_TYPES = ['five_prime_UTR','three_prime_UTR','CDS','UTR']
    
def get_id_table(path):
    id_convert = pd.read_csv(path,sep='\t').to_dict('list')
    table = {}
    for g_id,t_id in zip(id_convert['gene_id'],id_convert['transcript_id']):
        table[t_id] = g_id
    return table

def consist_(data,by,ref_value,drop_duplicated):
    returned = []
    ref_names = set(data[by])
    ref_names = [name for name in ref_names if str(name)!='nan']
    sectors = data.groupby(by)
    for name in ref_names:
        sector = sectors.get_group(name)
        max_value = max(sector[ref_value])
        sector = sector.to_dict('record')
        list_ = []
        for item in sector:
            if item[ref_value]==max_value:
                list_.append(item)
        true_count = len(list_)
        if drop_duplicated:
            if true_count==1:
                returned += list_
        else:
            returned += list_
    return returned

def consist(data,by,ref_value,drop_duplicated):
    strands = set(data['strand'])
    chrs = set(data['chr'])
    return_data = []
    for strand_ in strands:
        for chr_ in chrs:
            subdata = data[(data['strand'] == strand_) & (data['chr'] == chr_)]
            consist_data = consist_(subdata,by,ref_value,drop_duplicated)
            return_data+=consist_data
    df = pd.DataFrame.from_dict(return_data).drop_duplicates()
    return  df

def coordinate_consist_filter(data,group_by,site_name):
    returned = []
    ref_names = set(data[group_by])
    ref_names = [name for name in ref_names if str(name)!='nan']
    sectors = data.groupby(group_by)
    for name in ref_names:
        sector = sectors.get_group(name)
        value = set(list(sector[site_name]))
        if len(value) == 1:
            returned += sector.to_dict('record')
    return pd.DataFrame.from_dict(returned).drop_duplicates()

def duplicated_filter(data,group_by,site_name):
    """Drop all duplicate data in group"""
    returned = []
    ref_names = set(data[group_by])
    ref_names = [name for name in ref_names if str(name)!='nan']
    sectors = data.groupby(group_by)
    for name in ref_names:
        sector = sectors.get_group(name)
        sector = sector.drop_duplicates(site_name,keep=False)
        returned += sector.to_dict('record')
    return pd.DataFrame.from_dict(returned)

def classify_data_by_id(bed,selected_ids,id_convert=None):
    all_ids =list(set(bed['id']))
    selected_ids = set(selected_ids)
    if id_convert is not None:
        selected_gene_ids = set(id_convert[id_] for id_ in selected_ids)
        id_info = pd.DataFrame(all_ids)
        id_info.columns=['ref_id']
        gene_id = []
        for id_ in id_info['ref_id']:
            if id_ in id_convert.keys():
                gene_id.append(id_convert[id_])
            else:
                gene_id.append(id_)
        id_info = id_info.assign(gene_id=pd.Series(gene_id).values)
        match_status = id_info['gene_id'].isin(selected_gene_ids)
        want_status = id_info['ref_id'].isin(selected_ids)
        id_info['status'] = 'unwant'
        id_info.loc[match_status & want_status,'status'] = 'want'
        id_info.loc[match_status & ~want_status,'status'] = 'discard'
        want_id = id_info[id_info['status']=='want']['ref_id']
        unwant_id = id_info[id_info['status']=='unwant']['ref_id']
    else:
        id_info = pd.DataFrame(all_ids,columns=['id'])
        want_status = id_info['id'].isin(selected_ids)
        want_id = id_info[want_status]['id']
        unwant_id = id_info[~want_status]['id']
    want_bed = bed[bed['id'].isin(want_id)].drop_duplicates()
    unwant_bed = bed[bed['id'].isin(unwant_id)].drop_duplicates()
    return want_bed, unwant_bed

def simply_coord(bed):
    bed = bed[BED_COLUMNS[:6]]
    bed = bed.assign(id=pd.Series('.', index=bed.index))
    bed = bed.assign(score = pd.Series('.', index=bed.index))
    bed = bed.drop_duplicates()
    return bed

def simply_coord_with_gene_id(bed,id_convert=None):
    bed = bed[BED_COLUMNS[:6]]
    if id_convert is not None:
        gene_ids = [id_convert[id_] for id_ in bed['id']]
        bed = bed.assign(id=pd.Series(gene_ids).values)
    bed = bed.assign(score = pd.Series('.', index=bed.index))
    bed = bed.drop_duplicates()
    return bed

def _get_feature_coord(gff_item):
    part = [str(gff_item[type_]) for type_ in ['feature','chr','strand','start','end']]
    feature_coord = '_'.join(part)
    return feature_coord

def get_gff_with_intron(gff):
    gff = get_gff_with_attribute(gff)
    gff = get_gff_with_feature_coord(gff)
    ids = set(gff[gff['feature'].isin(RNA_TYPES)]['id'])
    exons = gff[gff['feature'].isin(EXON_TYPES)]
    exon_groups = exons.groupby('parent')
    exon_parents = set(exons['parent'])
    for id_ in ids:
        if id_ in exon_parents:
            item = gff[(gff['id']==id_) & (gff['feature'].isin(RNA_TYPES))].to_dict('record')[0]
            exon_group = exon_groups.get_group(id_).sort_values('start')
            exon_sites = sorted(exon_group['start'].tolist()+exon_group['end'].tolist())
            template = exon_group.to_dict('record')[0]
            for index in range(1,len(exon_sites)-1,2):
                template_ = dict(template)
                template_['feature'] = 'intron'
                template_['id'] = '{}_intron_{}'.format(template_['parent'],index)
                template_['start'] = exon_sites[index] + 1
                template_['end'] = exon_sites[index+1] - 1
                template_['attribute'] = 'ID={};Parent={}'.format(template_['id'],template_['parent'])
                feature_coord = _get_feature_coord(template_)
                template_['feature_coord'] = feature_coord
                if feature_coord not in set(gff['feature_coord']):
                    gff = gff.append(template_,ignore_index=True,verify_integrity=True)
    return gff[GFF_COLUMNS]

def get_gff_with_intergenic_region(gff,chrom_length):
    gff = get_gff_with_feature_coord(gff)
    genes = gff[gff['feature'].isin(GENE_TYPES)]
    for chrom_id,length in chrom_length.items():
        min_start=1
        max_end=length+min_start-1
        for strand in ['+','-']:
            template = {'chr':chrom_id,'strand':strand,'feature':'intergenic region'}
            items = genes[(genes['chr']==chrom_id) & (genes['strand']==strand)].sort_values('start')
            region_sites = sorted((items['start']-1).tolist()+(items['end']+1).tolist())
            if len(region_sites)>0:
                if min(region_sites) != min_start:
                    region_sites = [min_start] + region_sites
                if max(region_sites) != max_end:
                    region_sites = region_sites + [max_end]
            else:
                region_sites = [min_start,max_end]
            for index in range(0,len(region_sites)-1,2):
                gff_item = dict(template)
                gff_item['start'] = region_sites[index]
                gff_item['end'] = region_sites[index+1]
                feature_coord = _get_feature_coord(gff_item)
                gff_item['feature_coord'] = feature_coord
                for column in GFF_COLUMNS:
                    if column not in gff_item:
                        gff_item[column] = '.'
                if feature_coord not in set(gff['feature_coord']):
                    gff = gff.append(gff_item,ignore_index=True,verify_integrity=True)
    return gff[GFF_COLUMNS]

def gff_to_bed_command(gff_path,bed_path):
    to_bed_command = 'python3 {}/gff2bed.py -i {} -o {}'
    command = to_bed_command.format(preprocess_src_root,gff_path,bed_path)
    os.system(command)
