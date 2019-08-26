import os, sys
import pandas as pd
from argparse import ArgumentParser
from utils import consist, coordinate_consist_filter, duplicated_filter

def get_gene_names(df,name,convert_table):
    return [convert_table[name] for name in df[name]]

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will output sites data which are valid in RNA external UTR"+
                            ", and have no strogner signal in orphan site")
    parser.add_argument("--tg",help="transcript_gro_sites_path",required=True)
    parser.add_argument("--tc",help="transcript_cleavage_sites_path",required=True)
    parser.add_argument("--ig",help="inner_gro_sites_path",required=True)
    parser.add_argument("--ic",help="inner_cleavage_sites_path",required=True)
    parser.add_argument("--lg",help="long_dist_gro_sites_path",required=True)
    parser.add_argument("--lc",help="long_dist_cleavage_sites_path",required=True)
    parser.add_argument("-s", "--saved_root",help="saved_root",required=True)
    args = parser.parse_args()
    transcript_gro_sites_path = args.tg
    transcript_cleavage_sites_path = args.tc
    inner_gro_sites_path = args.ig
    inner_cleavage_sites_path = args.ic
    long_dist_gro_sites_path = args.lg
    long_dist_cleavage_sites_path = args.lc
    saved_root = args.saved_root
    safe_merged_gro_sites_path = os.path.join(saved_root,'safe_gro_sites.tsv')
    safe_merged_cleavage_sites_path = os.path.join(saved_root,'safe_cleavage_sites.tsv')
    exists = [os.path.exists(path) for path in [safe_merged_gro_sites_path,safe_merged_cleavage_sites_path]]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        print('Left only most significant signals')
        ###Read file###
        inner_gro_sites = pd.read_csv(inner_gro_sites_path,sep='\t')
        inner_ca_sites = pd.read_csv(inner_cleavage_sites_path,sep='\t')
        ld_gro_sites = pd.read_csv(long_dist_gro_sites_path,sep='\t')
        ld_ca_sites = pd.read_csv(long_dist_cleavage_sites_path,sep='\t')        
        transcript_gro_sites = pd.read_csv(transcript_gro_sites_path,sep='\t')
        transcript_ca_sites = pd.read_csv(transcript_cleavage_sites_path,sep='\t')
        #Create orhpan data
        ld_gro_id = set(ld_gro_sites['id'])
        ld_ca_id = set(ld_ca_sites['id'])
        inner_gro_id = set(inner_gro_sites['id'])
        inner_ca_id = set(inner_ca_sites['id'])
        orphan_ld_gro_id = ld_gro_id - inner_gro_id
        orphan_ld_ca_id = ld_ca_id - inner_ca_id
        orphan_ld_gro_sites = ld_gro_sites[ld_gro_sites['id'].isin(orphan_ld_gro_id)]
        orphan_ld_ca_sites = ld_ca_sites[ld_ca_sites['id'].isin(orphan_ld_ca_id)]
        #Create transcript-other-than-inner data
        #all_transcript_gro_sites = pd.concat([inner_gro_sites,transcript_gro_sites],sort=False)
        #other_gro_sites = duplicated_filter(all_transcript_gro_sites,'ref_name','evidence_5_end')
        #all_transcript_ca_sites = pd.concat([inner_ca_sites,transcript_ca_sites],sort=False)
        #other_ca_sites = duplicated_filter(all_transcript_ca_sites,'ref_name','evidence_3_end')
        #Merge data###
        merged_gro_sites = [inner_gro_sites,orphan_ld_gro_sites]#other_gro_sites
        merged_gro_sites = pd.concat(merged_gro_sites,sort=False)
        merged_ca_sites = [inner_ca_sites,orphan_ld_ca_sites]#other_ca_sites
        merged_ca_sites = pd.concat(merged_ca_sites,sort=False)
        print("Get consist site of every transcript")
        #Get max signal TSS
        merged_gro_sites = consist(merged_gro_sites,'ref_name','tag_count',True)
        #Preseve TSS located on external 5' exon
        consist_gro_site = merged_gro_sites[merged_gro_sites['gro_source'].isin(['inner'])]
        #Get max signal CA
        merged_ca_sites = consist(merged_ca_sites,'ref_name','read_count',True)
        #Preseve CA located on external 3' exon
        consist_ca_sites = merged_ca_sites[merged_ca_sites['cleavage_source'].isin(['inner'])]
        print('Export data')
        ###Write data###
        consist_gro_site.to_csv(safe_merged_gro_sites_path,sep='\t',index=False)
        consist_ca_sites.to_csv(safe_merged_cleavage_sites_path,sep='\t',index=False)
