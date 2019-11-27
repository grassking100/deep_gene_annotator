import sys,os
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
try:
    from sequence_annotation.sequence_annotation.utils.utils import BED_COLUMNS,GFF_COLUMNS
except:    
    from sequence_annotation.utils.utils import BED_COLUMNS,GFF_COLUMNS

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
        sites = sector[site_name]
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

def merge_bed_by_coord(bed):
    bed = bed[BED_COLUMNS[:6]]
    coord_ids = "{}_{}_{}_{}".format(bed['chr'],bed['strand'],bed['start'],bed['end'])
    bed = bed.assign(coord_id=pd.Series(coord_ids).values)
    coord_ids = set(coord_ids)
    groups = bed.groupby('coord_id')
    data = []
    for id_ in coord_ids:
        group = groups.get_group(id_)
        new_id = '_'.join(group['id'].astype(str))
        item = dict(group.to_dict('record')[0])
        del item['coord_id']
        item['id'] = new_id
        data+=[item]
    return pd.DataFrame.from_dict(data).drop_duplicates()
