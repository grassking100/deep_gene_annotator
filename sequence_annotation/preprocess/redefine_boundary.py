import os
import sys
import pandas as pd
from argparse import ArgumentParser
from multiprocessing import Pool
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_json, create_folder
from sequence_annotation.utils.stats import get_stats
from sequence_annotation.utils.get_intersection_id import intersection
from sequence_annotation.file_process.utils import GFF_COLUMNS, TRANSCRIPT_TYPE, read_bed, read_gff,write_gff
from sequence_annotation.file_process.utils import get_gff_with_updated_attribute, write_bed
from sequence_annotation.file_process.get_id_table import get_id_convert_dict
from sequence_annotation.file_process.coordinate_compare import coordinate_compare
from sequence_annotation.file_process.bed2gff import bed2gff
from sequence_annotation.file_process.site_analysis import get_site_diff


def plot_distance_hist(title, path, values):
    plt.figure()
    plt.hist(values, 100)
    plt.title(title)
    plt.xlabel("distance")
    plt.ylabel("number")
    plt.savefig(path)

def calculate_distance_between_sites(bed,TSS,CS,id_dict,output_root):
    create_folder(output_root)
    gff = bed2gff(bed,id_dict)
    gff = gff[gff['feature']==TRANSCRIPT_TYPE]
    abs_tss_site_e_to_r = get_site_diff(gff,TSS,multiprocess=40)
    abs_tss_site_r_to_e = get_site_diff(gff,TSS,answer_as_ref=False,multiprocess=40)
    abs_cleavage_site_e_to_r = get_site_diff(gff,CS,five_end=False,multiprocess=40)
    abs_cleavage_site_r_to_e = get_site_diff(gff,CS,five_end=False,answer_as_ref=False,multiprocess=40)
    tss_site_e_to_r = get_site_diff(gff,TSS,multiprocess=40,absolute=False)
    tss_site_r_to_e = get_site_diff(gff,TSS,answer_as_ref=False,multiprocess=40,absolute=False)
    cleavage_site_e_to_r = get_site_diff(gff,CS,five_end=False,multiprocess=40,absolute=False)
    cleavage_site_r_to_e = get_site_diff(gff,CS,five_end=False,answer_as_ref=False,multiprocess=40,absolute=False)
    
    stats_result = {}
    stats_result['tss_site_r_to_e'] = get_stats(tss_site_r_to_e)
    stats_result['tss_site_e_to_r'] = get_stats(tss_site_e_to_r)
    stats_result['cleavage_site_r_to_e'] = get_stats(cleavage_site_r_to_e)
    stats_result['cleavage_site_e_to_r'] = get_stats(cleavage_site_e_to_r)
    stats_result['abs_tss_site_r_to_e'] = get_stats(abs_tss_site_r_to_e)
    stats_result['abs_tss_site_e_to_r'] = get_stats(abs_tss_site_e_to_r)
    stats_result['abs_cleavage_site_r_to_e'] = get_stats(abs_cleavage_site_r_to_e)
    stats_result['abs_cleavage_site_e_to_r'] = get_stats(abs_cleavage_site_e_to_r)
    
    siter_path = os.path.join(output_root, "site.tsv")
    tss_path=os.path.join(output_root, 'tss_site_r_to_e.png')
    tss_ref_path=os.path.join(output_root, 'tss_site_e_to_r.png')
    cs_path=os.path.join(output_root, 'cleavage_site_r_to_e.png')
    cs_ref_path=os.path.join(output_root, 'cleavage_site_e_to_r.png')
    
    tss_title='The distance from the closest reference TSS to experimental TSS'
    tss_ref_title='The distance from the closest experimental TSS to reference TSS'
    cs_title='The distance from the closest reference CS to experimental CS'
    cs_ref_title='The distance from the closest experimental CS to reference CS'
    
    pd.DataFrame.from_dict(stats_result).T.to_csv(siter_path,index_label='stats',sep='\t')
    plot_distance_hist(tss_title,tss_path,tss_site_r_to_e)
    plot_distance_hist(tss_ref_title,tss_ref_path,tss_site_e_to_r)
    plot_distance_hist(cs_title,cs_path,cleavage_site_r_to_e)
    plot_distance_hist(cs_ref_title,cs_ref_path,cleavage_site_e_to_r)

def get_external_UTR(bed):
    external_UTR = []
    for item in bed.to_dict('record'):
        strand = item['strand']
        if strand not in ['+', '-']:
            raise Exception("Wrong strnad {}".format(strand))
        exon_starts,exon_ends = [], []
        thick_start,thick_end = item['thick_start'], item['thick_end']
        start,count  = item['start'],item['count']
        exon_sizes = [int(val) for val in item['block_size'].split(',')[:count]]
        exon_rel_starts = [int(val) for val in item['block_related_start'].split(',')[:count]]
        for exon_rel_start, exon_size in zip(exon_rel_starts, exon_sizes):
            exon_start = exon_rel_start + start
            exon_end = exon_start + exon_size - 1
            exon_starts.append(exon_start)
            exon_ends.append(exon_end)
        left_target_start, right_target_start = min(exon_starts), max(exon_starts)
        left_target_end = exon_ends[exon_starts.index(left_target_start)]
        right_target_end = exon_ends[exon_starts.index(right_target_start)]
        left,right = {}, {}
        right['id'] = left['id'] = item['id']
        right['strand'] = left['strand'] = strand
        right['chr'] = left['chr'] = item['chr']
        left.update({'start': left_target_start, 'end': left_target_end})
        right.update({'start': right_target_start, 'end': right_target_end})
        #if transcript is coding:
        if thick_start <= thick_end:
            left['end'] = min(left_target_end, thick_start - 1)
            right['start'] = max(right_target_start, thick_end + 1)
        if strand == '+':
            left['type'] = 'five_external_utr'
            right['type'] = 'three_external_utr'
        else:
            left['type'] = 'three_external_utr'
            right['type'] = 'five_external_utr'

        #If left UTR are exist
        if left['start'] <= left['end']:
            external_UTR.append(left)
        #If right UTR are exist
        if right['start'] <= right['end']:
            external_UTR.append(right)

    external_UTR = pd.DataFrame.from_dict(external_UTR)
    external_UTR['score'] = '.'
    return external_UTR


def _get_specific_utr(utr_bed, type_):
    utr_bed = utr_bed[utr_bed['type'].isin([type_])]
    utr_bed = utr_bed[['chr', 'start', 'end', 'id', 'score', 'strand']]
    return utr_bed


def get_five_external_utr(utr_bed):
    return _get_specific_utr(utr_bed, 'five_external_utr')


def get_three_external_utr(utr_bed):
    return _get_specific_utr(utr_bed, 'three_external_utr')

def _get_with_belong(data,name,lb,ub,belonging):
    returned = []
    if lb <= data[name] <= ub:
        temp = dict(data)
        temp['belonging'] = belonging
        returned.append(temp)
    return returned

def simple_belong_by_boundary(sites,boundaries,site_name,start_name,end_name,name_column):
    returned = []
    sites = sites.to_dict('record')
    boundaries = boundaries.to_dict('record')
    for boundary in boundaries:
        start, end = boundary[start_name], boundary[end_name]
        belonging = boundary[name_column]
        lb, ub = min(start,end), max(start,end)
        for site in sites:
            site_value = site[site_name]
            if lb <= site_value <= ub:
                temp = dict(site)
                temp['belonging'] = belonging
                returned.append(temp)
    return returned


def belong_by_boundary(sites,boundaries,exp_site_name,boundary_start_name,
                       boundary_end_name,name_column):
    strands,chrs = set(sites['strand']), set(sites['chr'])
    args_list = []
    for strand in strands:
        for chrom in chrs:
            print(chrom," ",strand)
            selected_s = sites[(sites['strand'] == strand) & (sites['chr'] == chrom)]
            selected_b = boundaries[(boundaries['strand'] == strand) & (boundaries['chr'] == chrom)]
            if len(selected_s) > 0 and len(selected_b) > 0:
                args_list.append((selected_s,selected_b,exp_site_name,boundary_start_name,
                                  boundary_end_name,name_column))
                
    with Pool(processes=len(strands)*len(chrs)) as pool:
        returned_ = pool.starmap(simple_belong_by_boundary,args_list)
    returned = []
    for item in returned_:
        returned += item
    df = pd.DataFrame.from_dict(returned)
    if df.empty:
        df = pd.DataFrame(columns = GFF_COLUMNS+['coord_id'])
    if any(df.duplicated()):
        raise
    df['attribute'] = None
    return  df


def belong_by_distance(sites,ref_sites,five_dist,three_dist,exp_site_name,ref_site_name,name_column):
    ub_name = '{}_ub'.format(ref_site_name)
    lb_name = '{}_lb'.format(ref_site_name)
    ref_sites = ref_sites.copy()
    ref_sites[ub_name] = ref_sites[lb_name] = None
    ref_plus_locs = ref_sites['strand'] == '+'
    ref_minus_locs = ref_sites['strand'] == '-'
    ref_values = ref_sites[ref_site_name]
    ref_sites.loc[ref_plus_locs,lb_name] = ref_values[ref_plus_locs] + five_dist
    ref_sites.loc[ref_plus_locs,ub_name] = ref_values[ref_plus_locs] + three_dist
    ref_sites.loc[ref_minus_locs,ub_name] = ref_values[ref_minus_locs] - five_dist
    ref_sites.loc[ref_minus_locs,lb_name] = ref_values[ref_minus_locs] - three_dist
    df = belong_by_boundary(sites,ref_sites,exp_site_name,lb_name,
                             ub_name,name_column)
    return df


def _consist(data, by, ref_value, drop_duplicated):
    returned = []
    for name,sector in data.groupby(by):
        max_value = max(sector[ref_value])
        sector = sector.to_dict('record')
        list_ = []
        for item in sector:
            if item[ref_value] == max_value:
                list_.append(item)
        true_count = len(list_)
        
        if drop_duplicated:
            if true_count == 1:
                returned += list_
        else:
            returned += list_
    return returned


def consist(data, by, ref_value, drop_duplicated):
    strands = set(data['strand'])
    chrs = set(data['chr'])
    returned = []
    for strand_ in strands:
        for chr_ in chrs:
            subdata = data[(data['strand'] == strand_) & (data['chr'] == chr_)]
            returned += _consist(subdata, by, ref_value, drop_duplicated)
    df = pd.DataFrame.from_dict(returned)
    return df

def get_coord_ids(gff):
    chroms = gff['chr']
    strands = gff['strand']
    starts = gff['start'].astype(str)
    coord_id = chroms + "_" + strands + "_" + starts
    return coord_id

def preprocess_exp_data(gff):
    coord_id = get_coord_ids(gff)
    gff = gff.assign(coord_id=coord_id)
    if gff['coord_id'].duplicated().any():
        raise
    gff['experimental_score'] = gff['experimental_score'].astype(float)
    return gff


def get_ophan_on_ld_data(on_long_dist, on_external_UTR):
    on_orphan  = on_long_dist[~on_long_dist['coord_id'].isin(on_external_UTR['coord_id'])].copy()
    on_orphan['feature'] = 'orphan'
    return on_orphan


def get_gff_with_coord_transcript_id(gff):
    gff = gff.assign(coord_transcript_id=gff['coord_id'] + "_" + gff['belonging'])
    return gff


def get_not_same_coord_transcript_id(compared, comparing):
    columns = compared.columns
    compared = get_gff_with_coord_transcript_id(compared)
    comparing = get_gff_with_coord_transcript_id(comparing)
    compared = compared[~compared['coord_transcript_id'].isin(comparing['coord_transcript_id'])]
    return compared.copy()[columns]

def get_transcrit_not_utr(on_transcript,on_external_UTR):
    other = get_not_same_coord_transcript_id(on_transcript, on_external_UTR)
    other['feature'] = 'other'
    other = get_gff_with_updated_attribute(other)
    return other

def get_consist_site(on_external_UTR, on_long_dist, on_transcript,remove_conflict=False):
    #Create orhpan data
    on_orphan = get_ophan_on_ld_data(on_long_dist, on_external_UTR)
    data = [on_external_UTR, on_orphan]
    if remove_conflict:
        #Add signal in transcript but not in external UTR to "other"
        other = get_not_same_coord_transcript_id(on_transcript, on_external_UTR)
        other['feature'] = 'other'
        other = get_gff_with_updated_attribute(other)
        data.append(other)
    data = pd.concat(data, sort=False).reset_index(drop=True)
    #Get consist site of every transcript
    consist_site = consist(data,'belonging','experimental_score',drop_duplicated=True)
    consist_site['attribute'] = None
    return consist_site


def coordinate_consist_filter(data,group_by,site_name):
    returned = []
    sectors = data.groupby(group_by)
    for name in set(data[group_by]):
        sector = sectors.get_group(name)
        value = set(list(sector[site_name]))
        if len(value) == 1:
            returned += sector.to_dict('record')
    return pd.DataFrame.from_dict(returned)


def get_coordinate(tss,cs,id_convert_dict,single_start_end=False):
    columns = ['experimental_score','start','source','feature','chr','strand','belonging']
    tss = tss[columns]
    cs = cs[columns]
    tss = tss.rename(columns={'experimental_score':'tss_score','start':'evidence_5_end',
                              'source':'tss_source','feature':'tss_feature'})
    cs = cs.rename(columns={'experimental_score':'cleavage_site_score','start':'evidence_3_end',
                            'source':'cleavage_site_source','feature':'cleavage_site_feature'})
    merged_data = cs.merge(tss,left_on=['chr','strand','belonging'],right_on=['chr','strand','belonging'])
    stats = {}
    returned = merged_data[(~merged_data['evidence_5_end'].isna()) & (~merged_data['evidence_3_end'].isna())]
    returned = returned.rename(columns={'belonging':'id'})
    stats['Boundary'] = len(returned)
    plus_index = returned['strand'] == '+'
    minus_index = returned['strand'] == '-'
    plus_valid_order = (returned.loc[plus_index,'evidence_5_end'] <= returned.loc[plus_index,'evidence_3_end']).index
    minus_valid_order = (returned.loc[minus_index,'evidence_5_end'] >= returned.loc[minus_index,'evidence_3_end']).index
    returned = returned.loc[list(plus_valid_order)+list(minus_valid_order),:]
    evidence_site = returned[['evidence_5_end','evidence_3_end']]
    returned['start'] =  evidence_site.min(1)
    returned['end'] =  evidence_site.max(1)
    returned['gene_id'] = [id_convert_dict[id_] for id_ in list(returned['id'])]
    stats['Transcript'] = len(returned)
    if single_start_end:
        returned = consist(returned,'gene_id','tss_score',drop_duplicated=False)
        returned = consist(returned,'gene_id','cleavage_site_score',drop_duplicated=False)
        returned = coordinate_consist_filter(returned,'gene_id','start')
        returned = coordinate_consist_filter(returned,'gene_id','end')
        stats['Transcript which its gene has single start and single end'] = len(returned)

    returned['source'] = 'Experiment'
    returned['feature'] = 'boundary'
    returned['score'] = returned['frame'] = '.'
    returned['attribute'] = None
    return returned, stats

def create_coordinate_bed(coordinate,bed,id_convert_dict,single_orf_start_end=False):
    coordinate = coordinate[['start', 'end', 'id']].copy()
    coordinate = coordinate.rename(columns={'start': 'coordinate_start','end': 'coordinate_end'})
    merged = bed.merge(coordinate,left_on='id',right_on='id')
    merged['start_diff'] = merged['coordinate_start'] - merged['start']
    merged['end_diff'] = merged['coordinate_end'] - merged['end']
    returned = []
    for item in merged.to_dict('record'):
        count = item['count']
        block_related_start = [int(val) for val in item['block_related_start'].split(',')[:count]]
        block_size = [int(val) for val in item['block_size'].split(',')[:count]]
        block_related_start = [start - item['start_diff'] for start in block_related_start]
        block_related_start[0] = 0
        block_size[0] -= item['start_diff']
        block_size[-1] += item['end_diff']
        for val in block_related_start:
            if val < 0:
                raise Exception("{} has negative relative start site {}".format(item['id'],val))
        for val in block_size:
            if val <= 0:
                raise Exception("{} has nonpositive size {}".format(item['id'],val))
        template = dict(item)
        template['block_related_start'] = ','.join(str(c) for c in block_related_start)
        template['block_size'] = ','.join(str(c) for c in block_size)
        template['start'] = template['coordinate_start']
        template['end'] = template['coordinate_end']
        returned.append(template)
    returned = pd.DataFrame.from_dict(returned)
    returned['gene_id'] = [id_convert_dict[id_] for id_ in returned['id']]
    if single_orf_start_end:
        returned = coordinate_consist_filter(returned,'gene_id','thick_start')
        returned = coordinate_consist_filter(returned,'gene_id', 'thick_end')
    returned = returned.sort_values(by=['chr', 'start','end', 'strand'])
    return returned


def get_location_count(consist_sites):
    count = consist_sites[['coord_id','feature']].drop_duplicates()['feature'].value_counts()
    return count.to_dict()


def get_coding_id(bed):
    bed = bed.copy()
    bed['coding_length'] = bed['thick_end'] - bed['thick_start'] + 1
    ids = set(bed[bed['coding_length']>0]['id'])
    return ids


def get_unique_coord_id_num(gff):
    return len(set(get_coord_ids(gff)))


def main(input_bed_path,id_table_path,tss_path,cs_path,upstream_dist,downstream_dist,
         output_root,single_start_end=False,remove_conflict=False):
    ###Read file###
    id_convert_dict = get_id_convert_dict(id_table_path)
    bed = read_bed(input_bed_path)
    tss = read_gff(tss_path,with_attr=True,valid_features=False)
    cleavage_site = read_gff(cs_path,with_attr=True,valid_features=False)
    tss = preprocess_exp_data(tss)
    cleavage_site = preprocess_exp_data(cleavage_site)
    coding_transcript_ids = get_coding_id(bed)
    calculate_distance_between_sites(bed,tss,cleavage_site,id_convert_dict,
                                     os.path.join(output_root,'site_diff'))
    #Get external UTR
    utr_bed = get_external_UTR(bed)
    external_five_UTR = get_five_external_utr(utr_bed)
    external_three_UTR = get_three_external_utr(utr_bed)
    write_bed(external_five_UTR,os.path.join(output_root,"external_five_UTR.bed"))
    write_bed(external_three_UTR,os.path.join(output_root,"external_three_UTR.bed"))
    ###Classify TSSs and cleavage sites and write data###
    ld_TSS = belong_by_distance(tss,bed,-upstream_dist,-1,'start','five_end','id')
    ld_CS = belong_by_distance(cleavage_site,bed,1,downstream_dist,"start",'three_end','id')
    external_5_UTR_TSS = belong_by_boundary(tss,external_five_UTR,'start','start','end','id')
    external_3_UTR_CS = belong_by_boundary(cleavage_site,external_three_UTR,'start','start','end','id')
    transcript_TSS = belong_by_boundary(tss,bed,'start','start','end','id')
    transcript_CS = belong_by_boundary(cleavage_site,bed,'start','start','end','id')
    external_5_UTR_TSS['feature'] = 'external_5_UTR_TSS'
    external_3_UTR_CS['feature'] = 'external_3_UTR_CS'
    transcript_TSS['feature'] = 'transcript_TSS'
    transcript_CS['feature'] = 'transcript_CS'
    ld_TSS['feature'] = 'long_dist_TSS'
    ld_CS['feature'] = 'long_dist_CS'
    consist_TSS = get_consist_site(external_5_UTR_TSS, ld_TSS, transcript_TSS, remove_conflict)
    consist_CS = get_consist_site(external_3_UTR_CS, ld_CS, transcript_CS,remove_conflict)  
    consist_TSS_on_external_5_UTR = consist_TSS[consist_TSS['feature']=="external_5_UTR_TSS"]
    consist_CS_on_external_3_UTR = consist_CS[consist_CS['feature']=="external_3_UTR_CS"]
    coordinate,transcript_num = get_coordinate(consist_TSS_on_external_5_UTR,consist_CS_on_external_3_UTR,
                                               id_convert_dict,single_start_end)
    redefined_bed = create_coordinate_bed(coordinate,bed,id_convert_dict)
    ###Write data###
    stats = {}
    stats['Reference transcripts'] = len(bed)
    stats['Redefined transcripts'] = len(redefined_bed)
    stats['Processed number'] = transcript_num
    stats['TSS evidence'] = get_unique_coord_id_num(tss)
    stats['CS evidence'] = get_unique_coord_id_num(cleavage_site)
    stats['TSS location'] = {}
    stats['CS location'] = {}
    stats['TSS location']['External 5\'UTR TSS'] = get_unique_coord_id_num(external_5_UTR_TSS)
    stats['CS location']['External 3\'UTR CS'] = get_unique_coord_id_num(external_3_UTR_CS)
    stats['TSS location']['Transcript TSS'] = get_unique_coord_id_num(transcript_TSS)
    stats['CS location']['Transcript CS'] = get_unique_coord_id_num(transcript_CS)
    stats['TSS location']['Intergenic TSS'] = get_unique_coord_id_num(ld_TSS)
    stats['CS location']['Intergenic CS'] = get_unique_coord_id_num(ld_CS) 
    stats['Stongest TSS location'] = get_location_count(consist_TSS)
    stats['Stongest CS location'] = get_location_count(consist_CS)
    write_json(stats, os.path.join(output_root,'boundary_stats.json'))
    coordinate_compare(bed,redefined_bed).to_csv(os.path.join(output_root,"coordinate_compared.csv"))
    ids_list = [coding_transcript_ids,set(consist_TSS_on_external_5_UTR['belonging']),set(consist_CS_on_external_3_UTR['belonging'])]
    id_names = ["Coding","including TSS","including CS"]
    intersection(ids_list,id_names,venn_path=os.path.join(output_root,'venn.png'))    
    write_bed(redefined_bed,os.path.join(output_root,"coordinate_redefined.bed"))
    write_gff(external_5_UTR_TSS,os.path.join(output_root,"external_5_UTR_TSS.gff3"),valid_features=False)
    write_gff(transcript_TSS,os.path.join(output_root,"transcript_TSS.gff3"),valid_features=False)
    write_gff(ld_TSS,os.path.join(output_root,"ld_TSS.gff3"),valid_features=False)
    write_gff(external_3_UTR_CS,os.path.join(output_root,"external_3_UTR_CS.gff3"),valid_features=False)
    write_gff(transcript_CS,os.path.join(output_root,"transcript_CS.gff3"),valid_features=False)
    write_gff(ld_CS,os.path.join(output_root,"ld_CS.gff3"),valid_features=False)
    

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_bed_path",help="Path of BED file",required=True)
    parser.add_argument("-t", "--tss_path",help="Path of TSS evidence file",required=True)
    parser.add_argument("-c", "--cs_path",help="Path of cleavage sites evidence file",required=True)
    parser.add_argument("-u", "--upstream_dist", type=int,help="upstream_dist",required=True)
    parser.add_argument("-d", "--downstream_dist", type=int,help="downstream_dist",required=True)
    parser.add_argument("-o", "--output_root",required=True)
    parser.add_argument("--id_table_path",required=True)
    parser.add_argument("--single_start_end",help="If it is selected, then only RNA data "+
                        "which have start sites and end sites with strongest signal in same gene"+
                        "will be saved",action='store_true')
    parser.add_argument("--remove_conflict",help='Remove between transcript and external UTR',action='store_true')
    args = parser.parse_args()
    main(**vars(args))
