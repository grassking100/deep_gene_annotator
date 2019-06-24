import pandas as pd

BED_COLUMNS = ['chr','start','end','id','score','strand','orf_start','orf_end','rgb','count','block_size','block_related_start']

def unique_site(data,by,ref_value):
    returned = []
    ref_names = set(data[by])
    ref_names = [name for name in ref_names if str(name)!='nan']
    sectors = data.groupby(by)
    for name in ref_names:
        sector = sectors.get_group(name)
        value = list(sector[ref_value])
        if len(set(value)) == 1:
            sector = sector.to_dict('record')
            returned.append(sector[0])
    df = pd.DataFrame.from_dict(returned)       
    return df

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
    df = pd.DataFrame.from_dict(return_data)
    return  df

def coordinate_consist_filter_(data,group_by,site_name):
    returned = []
    ref_names = set(data[group_by])
    ref_names = [name for name in ref_names if str(name)!='nan']
    sectors = data.groupby(group_by)
    for name in ref_names:
        sector = sectors.get_group(name)
        max_index = sector[site_name].idxmax()
        max_data = sector.loc[max_index]
        true_count = sum(sector[site_name] == max_data[site_name])
        if true_count==len(sector):
            returned += sector.to_dict('record')
    return returned

def coordinate_consist_filter(data,group_by,site_name):
    data['chr'] = data['chr'].astype(str)
    strands = set(data['strand'])
    chrs = set(data['chr'])
    return_data = []
    for strand_ in strands:
        for chr_ in chrs:
            subdata = data[(data['strand'] == strand_) & (data['chr'] == chr_)]
            if len(subdata)>0:
                consist_data = coordinate_consist_filter_(subdata,group_by,site_name)
                return_data+=consist_data
    return  pd.DataFrame.from_dict(return_data)

def simple_belong_by_boundary(exp_sites,boundarys,site_name,start_name,end_name,ref_name):
    data = []
    exp_sites = exp_sites.to_dict('record')
    boundarys = boundarys.to_dict('record')
    for boundary in boundarys:
        start = int(boundary[start_name])
        end = int(boundary[end_name])
        lb = min(start,end)
        ub = max(start,end)
        for exp_site in exp_sites:
            site = int(exp_site[site_name])
            if lb <= site and site <= ub:
                temp = dict(exp_site)
                temp['ref_name'] = boundary[ref_name]
                data.append(temp)
    return data

def simple_belong_by_distance(exp_sites,ref_sites,upstream_dist,downstream_dist,exp_site_name,ref_site_name,ref_name):
    exp_sites = exp_sites.to_dict('record')
    ref_sites = ref_sites.to_dict('record')
    data = []
    for ref_site in ref_sites:
        r_s = int(ref_site[ref_site_name])
        ub = r_s + upstream_dist
        db = r_s + downstream_dist
        for exp_site in exp_sites:
            e_s = int(exp_site[exp_site_name])
            if ub <= e_s and e_s <= db:
                temp = dict(exp_site)
                temp['ref_name'] = ref_site[ref_name]
                data.append(temp)
    return data

def belong_by_boundary(exp_sites,boundarys,exp_site_name,boundary_start_name,boundary_end_name,ref_name):
    exp_sites['chr'] = exp_sites['chr'].astype(str)
    boundarys['chr'] = boundarys['chr'].astype(str)
    strands = set(exp_sites['strand'])
    chrs = set(exp_sites['chr'])
    returned_data = []
    for strand_ in strands:
        for chr_ in chrs:
            print(chr_," ",strand_)
            selected_exp_sites = exp_sites[(exp_sites['strand'] == strand_) & (exp_sites['chr'] == chr_)]
            selected_boundarys = boundarys[(boundarys['strand'] == strand_) & (boundarys['chr'] == chr_)]
            if len(selected_exp_sites)>0 and len(selected_boundarys)>0:
                selected_exp_site = simple_belong_by_boundary(selected_exp_sites,selected_boundarys,
                                                             exp_site_name,boundary_start_name,
                                                             boundary_end_name,ref_name)
                returned_data += selected_exp_site
    df = pd.DataFrame.from_dict(returned_data)
    return  df

def belong_by_distance(exp_sites,ref_sites,five_dist,three_dist,exp_site_name,ref_site_name,ref_name):
    exp_sites['chr'] = exp_sites['chr'].astype(str)
    ref_sites['chr'] = ref_sites['chr'].astype(str)
    returned_data = []
    strands = set(exp_sites['strand'])
    chrs = set(exp_sites['chr'])
    for strand_ in strands:
        if strand_ == '+':
            upstream_dist = five_dist
            downstream_dist = three_dist
        else:
            upstream_dist = -three_dist
            downstream_dist = -five_dist
        for chr_ in chrs:
            print(chr_," ",strand_)
            selected_exp_sites = exp_sites[(exp_sites['strand'] == strand_) & (exp_sites['chr'] == chr_)]
            selected_ref_sites = ref_sites[(ref_sites['strand'] == strand_) & (ref_sites['chr'] == chr_)]
            if len(selected_exp_sites)>0 and len(selected_ref_sites)>0:
                selected_exp_site = simple_belong_by_distance(selected_exp_sites,
                                                           selected_ref_sites,
                                                           upstream_dist,downstream_dist,
                                                           exp_site_name,ref_site_name,
                                                           ref_name)
                returned_data += selected_exp_site
    df = pd.DataFrame.from_dict(returned_data)
    return  df

def get_id_table(path):
    id_convert = pd.read_csv(path,sep='\t').to_dict('list')
    table = {}
    for g_id,t_id in zip(id_convert['gene_id'],id_convert['transcript_id']):
        table[t_id] = g_id
    return table

def get_bed_most_UTR(bed):
    most_UTR_site = []
    for item in bed.to_dict('record'):
        exon_starts = []
        exon_ends = []
        orf_start = int(item['orf_start'])
        orf_end = int(item['orf_end'])
        start = int(item['start'])
        count = int(item['count'])
        exon_sizes = [int(val) for val in item['block_size'].split(',')[:count]]
        exon_rel_starts = [int(val) for val in item['block_related_start'].split(',')[:count]]
        for exon_rel_start,exon_size in zip(exon_rel_starts,exon_sizes):
            exon_start = exon_rel_start + start
            exon_end = exon_start + exon_size - 1
            exon_starts.append(exon_start)
            exon_ends.append(exon_end)
        left_target_start = min(exon_starts)
        left_target_end = exon_ends[exon_starts.index(left_target_start)]
        right_target_start = max(exon_starts)
        right_target_end = exon_ends[exon_starts.index(right_target_start)]
        strand = item['strand']
        if strand == '+':
            five = {'start':left_target_start,'end':min(left_target_end,orf_start-1),'type':'five_most_utr'}
            three = {'start':max(right_target_start,orf_end+1),'end':right_target_end,'type':'three_most_utr'}
        else:
            three = { 'start':left_target_start,'end':min(left_target_end,orf_start-1),'type':'three_most_utr'}
            five = {'start':max(right_target_start,orf_end+1),'end':right_target_end,'type':'five_most_utr'}
        three['id'] = five['id'] = item['id']
        three['strand'] = five['strand'] = strand
        three['chr'] = five['chr'] = item['chr']
        if five['start'] <= five['end']:
            most_UTR_site.append(five)
        if three['start'] <= three['end']:
            most_UTR_site.append(three)
    most_UTR_site = pd.DataFrame.from_dict(most_UTR_site)
    return most_UTR_site

def classify_data_by_id(all_bed,selected_ids,id_convert=None):
    all_ids =list(set(all_bed['id']))
    selected_ids = set(selected_ids)
    if id_convert is not None:
        selected_gene_ids = set([id_convert[id_] for id_ in selected_ids])
        id_info = pd.DataFrame(all_ids)
        id_info.columns=['ref_id']
        gene_id = [id_convert[id_] for id_ in id_info['ref_id']]
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
    want_bed = all_bed[all_bed['id'].isin(want_id)]
    unwant_bed = all_bed[all_bed['id'].isin(unwant_id)]
    return want_bed, unwant_bed

def create_coordinate_bed(consist_data,valid_official_gene_info):
    coordinate_consist_data = valid_official_gene_info.merge(consist_data,left_on='id', right_on='ref_name')
    coordinate_consist_data['start_diff'] = coordinate_consist_data['coordinate_start'] - coordinate_consist_data['start']
    coordinate_consist_data['end_diff'] = coordinate_consist_data['coordinate_end'] - coordinate_consist_data['end']
    new_data = []
    for item in coordinate_consist_data.to_dict('record'):
        count = int(item['count'])
        block_related_start = [int(val) for val in item['block_related_start'].split(',')[:count]]
        block_size = [int(val) for val in item['block_size'].split(',')[:count]]
        start_diff = int(item['start_diff'])
        end_diff = int(item['end_diff'])
        block_related_start = [start-start_diff for start in block_related_start]
        block_related_start[0] = 0
        block_size[0] -= start_diff
        block_size[-1] += end_diff
        for val in block_related_start:
            if val<0:
                raise Exception(item['ref_name']+" has negative relative start site,"+str(val))
        for val in block_size:
            if val<=0:
                raise Exception(item['ref_name']+" has nonpositive size,"+str(val))
        temp = dict(item)
        temp['block_related_start'] = ','.join(str(c) for c in block_related_start)
        temp['block_size'] = ','.join(str(c) for c in block_size)
        temp['start'] = temp['coordinate_start']
        temp['end'] = temp['coordinate_end']
        new_data.append(temp)
    return pd.DataFrame.from_dict(new_data)

def read_bed(path):
    #Read bed data into pandas DataFrame format
    bed = pd.read_csv(path,sep='\t',header=None)
    bed.columns = BED_COLUMNS[:len(bed.columns)]
    data = []
    for item in bed.to_dict('record'):
        temp = dict(item)
        if temp['strand'] == '+':
            temp['five_end'] = temp['start'] + 1
            temp['three_end'] = temp['end']
        else:
            temp['five_end'] = temp['end']
            temp['three_end'] = temp['start'] + 1
        data.append(temp)
    df = pd.DataFrame.from_dict(data)
    df = df.astype(str)
    for name in ['start','end','orf_start','orf_end','count']:
        if name in df.columns:
            df[name] = df[name].astype(float).astype(int)
            if name in ['start','orf_start']:
                df[name] = df[name] + 1
    return df

def write_bed(bed,path):
    columns = []
    for column in BED_COLUMNS:
        if column in bed.columns:
            columns.append(column)
    bed = bed[columns]
    bed = bed.astype(str)
    for name in ['start','end','orf_start','orf_end','count']:
        if name in bed.columns:
            bed[name] = bed[name].astype(float).astype(int)
            if name in ['start','orf_start']:
                bed[name] = bed[name] - 1
    bed.to_csv(path,sep='\t',index=None,header=None)

def simply_coord(bed):
    bed = bed[BED_COLUMNS[:6]]
    bed['id'] = '.'
    bed['score'] = '.'
    bed = bed.drop_duplicates()
    return bed

def simply_coord_with_gene_id(bed,id_convert=None):
    bed = bed[BED_COLUMNS[:6]]
    if id_convert is not None:
        gene_ids = [id_convert[id_] for id_ in bed['id']]
        bed = bed.assign(id=pd.Series(gene_ids).values)
    bed['score'] = '.'
    bed = bed.drop_duplicates()
    return bed

def merge_bed_by_coord(bed):
    bed = bed[BED_COLUMNS[:6]]
    coord_ids = bed['chr'].astype(str)+"_"+bed['strand'].astype(str)+"_"+bed['start'].astype(str)+"_"+bed['end'].astype(str)
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
    return pd.DataFrame.from_dict(data)

def get_df_with_seq_id(df):
    df_dict = df.to_dict('record')
    data = []
    for item in df_dict:
        ids = item[8].split(';')
        type_ = item[2]
        item_id = ""
        item_name = ""
        item_parent = ""
        for id_ in ids:
            if id_.startswith("Parent"):
                item_parent = id_[7:]
            if id_.startswith("ID"):
                item_id = id_[3:]
            if id_.startswith("Name"):
                item_name = id_[5:]
        item_parent = item_parent.split(",")
        item_id = item_id.split(",")
        for parent in item_parent:
            for id_ in item_id:
                copied = dict(item)
                copied['id'] = id_
                copied['parent'] = parent
                copied['name'] = item_name
                data.append(copied)
    df = pd.DataFrame.from_dict(data)
    return df
