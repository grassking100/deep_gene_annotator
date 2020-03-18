import os, sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_gff,read_gff
from sequence_annotation.utils.utils import get_gff_with_updated_attribute,get_gff_with_attribute

def simple_belong_by_boundary(exp_sites,boundarys,exp_site_name,start_name,end_name,ref_site_name):
    data = []
    exp_sites = exp_sites.copy()
    boundarys = boundarys.copy()
    boundarys[start_name] = boundarys[start_name].astype(int)
    boundarys[end_name] = boundarys[end_name].astype(int)
    exp_sites[exp_site_name] = exp_sites[exp_site_name].astype(int)
    exp_sites = exp_sites.to_dict('record')
    boundarys = boundarys.to_dict('record')
    for boundary in boundarys:
        start, end = boundary[start_name], boundary[end_name]
        lb, ub = min(start,end), max(start,end)
        for exp_site in exp_sites:
            site = exp_site[exp_site_name]
            if lb <= site <= ub:
                temp = dict(exp_site)
                temp['ref_name'] = boundary[ref_site_name]
                data.append(temp)
    return data

def simple_belong_by_distance(exp_sites,ref_sites,upstream_dist,downstream_dist,
                              exp_site_name,ref_site_name,ref_name):
    data = []
    exp_sites = exp_sites.copy()
    ref_sites = ref_sites.copy()
    exp_sites[exp_site_name] = exp_sites[exp_site_name].astype(int)
    ref_sites[ref_site_name] = ref_sites[ref_site_name].astype(int)
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
    for strand in strands:
        for chrom in chrs:
            print(chrom," ",strand)
            selected_exp_sites = exp_sites[(exp_sites['strand'] == strand) & (exp_sites['chr'] == chrom)]
            selected_boundarys = boundarys[(boundarys['strand'] == strand) & (boundarys['chr'] == chrom)]
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
        raise InvalidStrandType(strands)

    chrs = set(exp_sites['chr'])
    for strand in strands:
        if strand == '+':
            upstream_dist = five_dist
            downstream_dist = three_dist
        else:
            upstream_dist = -three_dist
            downstream_dist = -five_dist
        for chrom in chrs:
            print(chrom," ",strand)
            selected_exp_sites = exp_sites[(exp_sites['strand'] == strand) & (exp_sites['chr'] == chrom)]
            selected_ref_sites = ref_sites[(ref_sites['strand'] == strand) & (ref_sites['chr'] == chrom)]
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
    parser.add_argument("-o", "--official_bed_path",help="Path of selected official gene info file",required=True)
    parser.add_argument("-g", "--tss_path",help="Path of selected TSSs file",required=True)
    parser.add_argument("-c", "--cleavage_site_path",help="Path of cleavage sites file",required=True)
    parser.add_argument("-f", "--external_five_UTR_path",help="Path of external five UTR file",required=True)
    parser.add_argument("-t", "--external_three_UTR_path",help="Path of external three UTR file",required=True)
    parser.add_argument("-s", "--saved_root",help="Path to save",required=True)
    parser.add_argument("-u", "--upstream_dist", type=int,help="upstream_dist",required=True)
    parser.add_argument("-d", "--downstream_dist", type=int,help="downstream_dist",required=True)
    args = parser.parse_args()
    external_5_UTR_TSS_path = os.path.join(args.saved_root,'external_five_UTR_tss.gff3')
    external_3_UTR_cleavage_site_path = os.path.join(args.saved_root,'external_three_UTR_cleavage_site.gff3')
    long_dist_tss_path = os.path.join(args.saved_root,'long_dist_tss.gff3')
    long_dist_cleavage_site_path = os.path.join(args.saved_root,'long_dist_cleavage_site.gff3')
    transcript_tss_path = os.path.join(args.saved_root,'transcript_tss.gff3')
    transcript_cleavage_site_path = os.path.join(args.saved_root,'transcript_cleavage_site.gff3')
    paths = [external_5_UTR_TSS_path,external_3_UTR_cleavage_site_path,
             long_dist_tss_path,long_dist_cleavage_site_path,
             transcript_tss_path,transcript_cleavage_site_path]
    exists = [os.path.exists(path) for path in paths]
    if all(exists):
         print("Result files are already exist, procedure will be skipped.")
    else:
        print('Find TSS and CA sites are belong to which genes')
        ###Read file###
        official_bed = read_bed(args.official_bed_path)
        tss = get_gff_with_attribute(read_gff(args.tss_path))
        cleavage_site = get_gff_with_attribute(read_gff(args.cleavage_site_path))
        external_five_UTR = read_bed(args.external_five_UTR_path)
        external_three_UTR = read_bed(args.external_three_UTR_path)
        
        print('Classify tss sites and cleavage sites and write data')
        ###Classify TSSs and cleavage sites and write data###
        external_5_UTR_TSS = belong_by_boundary(tss,external_five_UTR,'start','start','end','id')
        external_3_UTR_cleavage_site = belong_by_boundary(cleavage_site,external_three_UTR,'start','start','end','id')
        transcript_tss = belong_by_boundary(tss,official_bed,'start','start','end','id')
        transcript_cleavage_site = belong_by_boundary(cleavage_site,official_bed,'start','start','end','id')
        long_dist_tss = belong_by_distance(tss,official_bed,-args.upstream_dist,-1,'start','five_end','id')
        long_dist_cleavage_site = belong_by_distance(cleavage_site,official_bed,1,args.downstream_dist,"start",'three_end','id')
        
        external_5_UTR_TSS['feature'] = 'external_5_UTR_TSS'
        external_3_UTR_cleavage_site['feature'] = 'external_3_UTR_CS'
        transcript_tss['feature'] = 'transcript_TSS'
        transcript_cleavage_site['feature'] = 'transcript_CS'
        long_dist_tss['feature'] = 'long_dist_TSS'
        long_dist_cleavage_site['feature'] = 'long_dist_CS'
        ###Write data###
        tss = [external_5_UTR_TSS,transcript_tss,long_dist_tss]
        tss_paths = [external_5_UTR_TSS_path,transcript_tss_path,long_dist_tss_path]
        cs_sites = [external_3_UTR_cleavage_site,transcript_cleavage_site,long_dist_cleavage_site]
        cs_paths = [external_3_UTR_cleavage_site_path,transcript_cleavage_site_path,long_dist_cleavage_site_path]
        for tss,path in zip(tss,tss_paths):
            tss = get_gff_with_updated_attribute(tss)
            write_gff(tss,path)

        for cs,path in zip(cs_sites,cs_paths):
            cs = get_gff_with_updated_attribute(cs)
            write_gff(cs,path)
