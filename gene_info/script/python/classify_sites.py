import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__))
from utils import belong_by_boundary, belong_by_distance, read_bed
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
    parser.add_argument("-u", "--upstream_dist",
                        help="upstream_dist",required=True)
    parser.add_argument("-d", "--downstream_dist",
                        help="downstream_dist",required=True)
    parser.add_argument("-p", "--tolerate_dist",
                        help="tolerate_dist",required=True)
    args = vars(parser.parse_args())
    valid_official_bed_path = args['valid_official_bed_path']
    valid_gro_site_path = args['valid_gro_site_path']
    valid_cleavage_site_path = args['valid_cleavage_site_path']
    valid_external_five_UTR_path = args['valid_external_five_UTR_path']
    valid_external_three_UTR_path = args['valid_external_three_UTR_path']
    saved_root = args['saved_root']
    upstream_dist = int(args['upstream_dist'])
    downstream_dist = int(args['downstream_dist'])
    tolerate_dist = int(args['tolerate_dist'])
    dist_gro_sites_path = saved_root+'/dist_gro_sites'+'.tsv'
    dist_cleavage_sites_path = saved_root+'/dist_cleavage_sites'+'.tsv'
    inner_gro_sites_path = saved_root+'/inner_gro_sites'+'.tsv'
    inner_cleavage_sites_path = saved_root+'/inner_cleavage_sites'+'.tsv'
    long_dist_gro_sites_path = saved_root+'/long_dist_gro_sites'+'.tsv'
    long_dist_cleavage_sites_path = saved_root+'/long_dist_cleavage_sites'+'.tsv'
    orf_inner_gro_sites_path = saved_root+'/orf_inner_gro_sites'+'.tsv'
    orf_inner_cleavage_sites_path = saved_root+'/orf_inner_cleavage_sites'+'.tsv'
    print('Find TSS and CA sites are belong to which genes')
    paths = [dist_gro_sites_path,dist_cleavage_sites_path,
             inner_gro_sites_path,inner_cleavage_sites_path,
             long_dist_gro_sites_path,long_dist_cleavage_sites_path,
             orf_inner_gro_sites_path,orf_inner_cleavage_sites_path]
    exists = [os.path.exists(path) for path in paths]
    if all(exists):
         print("Result files are already exist, procedure will be skipped.")
    else:
        ###Read file###
        valid_official_bed = read_bed(valid_official_bed_path,mark_with_one_based_sites=True)
        valid_gro = pd.read_csv(valid_gro_site_path,sep='\t')
        valid_cleavage_site = pd.read_csv(valid_cleavage_site_path,sep='\t')
        valid_external_five_UTR = pd.read_csv(valid_external_five_UTR_path,sep='\t')
        #valid_external_five_UTR['start'] += 1
        valid_external_three_UTR = pd.read_csv(valid_external_three_UTR_path,sep='\t')
        #valid_external_three_UTR['start'] += 1
        print('Classify valid GRO sites and cleavage sites and write data')
        ###Classify valid GRO sites and cleavage sites and write data###
        dist_gro_sites = belong_by_distance(valid_gro,valid_official_bed,
                                             -tolerate_dist,-1,
                                            'evidence_5_end','five_end','id')
        dist_cleavage_sites = belong_by_distance(valid_cleavage_site,valid_official_bed,
                                                  1,tolerate_dist,
                                                 "evidence_3_end",'three_end','id')
        inner_gro_sites = belong_by_boundary(valid_gro,valid_external_five_UTR,
                                               'evidence_5_end','start','end','id')
        inner_cleavage_sites = belong_by_boundary(valid_cleavage_site,valid_external_three_UTR,
                                                   'evidence_3_end','start','end','id')
        long_dist_gro_sites = belong_by_distance(valid_gro,valid_official_bed,
                                                 -upstream_dist,-(tolerate_dist+1),
                                                 'evidence_5_end','five_end','id')
        long_dist_cleavage_sites = belong_by_distance(valid_cleavage_site,valid_official_bed,
                                                      tolerate_dist+1,downstream_dist,
                                                      "evidence_3_end",'three_end','id')
        orf_inner_gro_sites = belong_by_boundary(valid_gro,valid_official_bed,
                                                 'evidence_5_end','orf_start','orf_end','id')
        orf_inner_cleavage_sites = belong_by_boundary(valid_cleavage_site,valid_official_bed,
                                                      'evidence_3_end','orf_start','orf_end','id')
        ###Write data###
        gro_columns = ['chr','strand','evidence_5_end','tag_count','ref_name']
        cs_columns = ['chr','strand','evidence_3_end','read_count','ref_name']
        gro_sites = [dist_gro_sites,inner_gro_sites,long_dist_gro_sites,orf_inner_gro_sites]
        gro_paths = [dist_gro_sites_path,inner_gro_sites_path,long_dist_gro_sites_path,orf_inner_gro_sites_path]
        for gro,path in zip(gro_sites,gro_paths):
            if len(gro)==0:
                 gro = pd.DataFrame(columns=gro_columns)
            gro = gro[gro_columns]
            gro.to_csv(path,sep='\t',index=False)
        cs_sites = [dist_cleavage_sites,inner_cleavage_sites,long_dist_cleavage_sites,orf_inner_cleavage_sites]
        cs_paths = [dist_cleavage_sites_path,inner_cleavage_sites_path,long_dist_cleavage_sites_path,orf_inner_cleavage_sites_path]
        for cs,path in zip(cs_sites,cs_paths):
            if len(cs)==0:
                 cs = pd.DataFrame(columns=cs_columns)
            cs = cs[cs_columns]
            cs.to_csv(path,sep='\t',index=False)
