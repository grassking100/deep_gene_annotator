import os, sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed

def simple_belong_by_boundary(exp_sites,boundarys,site_name,start_name,end_name,ref_name):
    data = []
    boundarys.loc[:,start_name] = boundarys[start_name].astype(int)
    boundarys.loc[:,end_name] = boundarys[end_name].astype(int)
    exp_sites.loc[:,site_name] = exp_sites[site_name].astype(int)
    exp_sites = exp_sites.to_dict('record')
    boundarys = boundarys.to_dict('record')
    for boundary in boundarys:
        start, end = boundary[start_name], boundary[end_name]        
        lb, ub = min(start,end), max(start,end)
        for exp_site in exp_sites:
            site = exp_site[site_name]
            if lb <= site <= ub:
                temp = dict(exp_site)
                temp['ref_name'] = boundary[ref_name]
                data.append(temp)
    return data

def simple_belong_by_distance(exp_sites,ref_sites,upstream_dist,downstream_dist,
                              exp_site_name,ref_site_name,ref_name):
    data = []
    ref_sites.loc[:,ref_site_name] = ref_sites[ref_site_name].astype(int)
    exp_sites.loc[:,exp_site_name] = exp_sites[exp_site_name].astype(int)
    exp_sites = exp_sites.to_dict('record')
    ref_sites = ref_sites.to_dict('record')

    for ref_site in ref_sites:
        r_s = ref_site[ref_site_name]
        ub = r_s + upstream_dist
        db = r_s + downstream_dist
        for exp_site in exp_sites:
            e_s = exp_site[exp_site_name]
            if ub <= e_s <= db:
                temp = dict(exp_site)
                temp['ref_name'] = ref_site[ref_name]
                data.append(temp)
    return data

def belong_by_boundary(exp_sites,boundarys,exp_site_name,boundary_start_name,
                       boundary_end_name,ref_name):
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
            if len(selected_exp_sites) > 0 and len(selected_boundarys) > 0:
                selected_exp_site = simple_belong_by_boundary(selected_exp_sites,selected_boundarys,
                                                              exp_site_name,boundary_start_name,
                                                              boundary_end_name,ref_name)
                returned_data += selected_exp_site
    df = pd.DataFrame.from_dict(returned_data).drop_duplicates()
    return  df

def belong_by_distance(exp_sites,ref_sites,five_dist,three_dist,exp_site_name,ref_site_name,ref_name):
    exp_sites['chr'] = exp_sites['chr'].astype(str)
    ref_sites['chr'] = ref_sites['chr'].astype(str)
    returned_data = []
    strands = set(exp_sites['strand'])
    if strands != set(['+','-']):
        raise Exception("Invalid strand in {}".format(strands))
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
            if len(selected_exp_sites) > 0 and len(selected_ref_sites) > 0:
                selected_exp_site = simple_belong_by_distance(selected_exp_sites,selected_ref_sites,
                                                              upstream_dist,downstream_dist,
                                                              exp_site_name,ref_site_name,ref_name)
                returned_data += selected_exp_site
    df = pd.DataFrame.from_dict(returned_data).drop_duplicates()
    return  df

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-o", "--valid_official_bed_path",
                        help="Path of selected official gene info file",required=True)
    parser.add_argument("-g", "--valid_gro_site_path",
                        help="Path of selected GRO sites file",required=True)
    parser.add_argument("-c", "--valid_cleavage_site_path",
                        help="Path of cleavage sites file",required=True)
    parser.add_argument("-f", "--valid_external_five_UTR_path",
                        help="Path of valid external five UTR file",required=True)
    parser.add_argument("-t", "--valid_external_three_UTR_path",
                        help="Path of valid external three UTR file",required=True)
    parser.add_argument("-s", "--saved_root",
                        help="Path to save",required=True)
    parser.add_argument("-u", "--upstream_dist", type=int,
                        help="upstream_dist",required=True)
    parser.add_argument("-d", "--downstream_dist", type=int,
                        help="downstream_dist",required=True)
    args = parser.parse_args()
    inner_gro_sites_path = os.path.join(args.saved_root,'inner_gro_sites.tsv')
    inner_cleavage_sites_path = os.path.join(args.saved_root,'inner_cleavage_sites.tsv')
    long_dist_gro_sites_path = os.path.join(args.saved_root,'long_dist_gro_sites.tsv')
    long_dist_cleavage_sites_path = os.path.join(args.saved_root,'long_dist_cleavage_sites.tsv')
    transcript_gro_sites_path = os.path.join(args.saved_root,'transcript_gro_sites.tsv')
    transcript_cleavage_sites_path = os.path.join(args.saved_root,'transcript_cleavage_sites.tsv')
    paths = [inner_gro_sites_path,inner_cleavage_sites_path,
             long_dist_gro_sites_path,long_dist_cleavage_sites_path,
             transcript_gro_sites_path,transcript_cleavage_sites_path]
    exists = [os.path.exists(path) for path in paths]
    if all(exists):
         print("Result files are already exist, procedure will be skipped.")
    else:
        print('Find TSS and CA sites are belong to which genes')
        ###Read file###
        valid_official_bed = read_bed(args.valid_official_bed_path)
        valid_gro = pd.read_csv(args.valid_gro_site_path,sep='\t')
        valid_cleavage_site = pd.read_csv(args.valid_cleavage_site_path,sep='\t')
        valid_external_five_UTR = pd.read_csv(args.valid_external_five_UTR_path,sep='\t')
        valid_external_three_UTR = pd.read_csv(args.valid_external_three_UTR_path,sep='\t')
        print('Classify valid GRO sites and cleavage sites and write data')
        ###Classify valid GRO sites and cleavage sites and write data###
        inner_gro_sites = belong_by_boundary(valid_gro,valid_external_five_UTR,
                                             'evidence_5_end','start','end','id')
        inner_cleavage_sites = belong_by_boundary(valid_cleavage_site,valid_external_three_UTR,
                                                 'evidence_3_end','start','end','id')
        transcript_gro_sites = belong_by_boundary(valid_gro,valid_official_bed,
                                                  'evidence_5_end','start','end','id')
        transcript_cleavage_sites = belong_by_boundary(valid_cleavage_site,valid_official_bed,
                                                       'evidence_3_end','start','end','id')
        long_dist_gro_sites = belong_by_distance(valid_gro,valid_official_bed,
                                                 -args.upstream_dist,-1,'evidence_5_end','five_end','id')
        long_dist_cleavage_sites = belong_by_distance(valid_cleavage_site,valid_official_bed,
                                                      1,args.downstream_dist,"evidence_3_end",'three_end','id')
        
        inner_gro_sites['gro_source'] = 'inner'
        inner_cleavage_sites['cleavage_source'] = 'inner'
        transcript_gro_sites['gro_source'] = 'transcript'
        transcript_cleavage_sites['cleavage_source'] = 'transcript'
        long_dist_gro_sites['gro_source'] = 'long_dist'
        long_dist_cleavage_sites['cleavage_source'] = 'long_dist'
        ###Write data###
        gro_columns = ['chr','strand','evidence_5_end','tag_count','ref_name','gro_source','id']
        cs_columns = ['chr','strand','evidence_3_end','read_count','ref_name','cleavage_source','id']
        gro_sites = [inner_gro_sites,transcript_gro_sites,long_dist_gro_sites]
        gro_paths = [inner_gro_sites_path,transcript_gro_sites_path,long_dist_gro_sites_path]
        cs_sites = [inner_cleavage_sites,transcript_cleavage_sites,long_dist_cleavage_sites]
        cs_paths = [inner_cleavage_sites_path,transcript_cleavage_sites_path,long_dist_cleavage_sites_path]
        for gro,path in zip(gro_sites,gro_paths):
            if len(gro)==0:
                 gro = pd.DataFrame(columns=gro_columns)
            gro = gro[gro_columns]
            gro.to_csv(path,sep='\t',index=False)
        for cs,path in zip(cs_sites,cs_paths):
            if len(cs)==0:
                 cs = pd.DataFrame(columns=cs_columns)
            cs = cs[cs_columns]
            cs.to_csv(path,sep='\t',index=False)
