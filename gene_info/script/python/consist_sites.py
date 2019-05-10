import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import consist,unique_site
import pandas as pd
from argparse import ArgumentParser
def get_gene_names(df,name,convert_table):
    return [convert_table[name] for name in df[name]]
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will used mRNA_bed12 data to create annotated genome\n"+
                            "and it will selecte region from annotated genome according to the\n"+
                            "selected_region and selected region will save to output_path in h5 format")
    parser.add_argument("--tg",
                        help="transcript_gro_sites_path",required=True)
    parser.add_argument("--tc",
                        help="transcript_cleavage_sites_path",required=True)
    parser.add_argument("--ig",
                        help="inner_gro_sites_path",required=True)
    parser.add_argument("--ic",
                        help="inner_cleavage_sites_path",required=True)
    parser.add_argument("--lg",
                        help="long_dist_gro_sites_path",required=True)
    parser.add_argument("--lc",
                        help="long_dist_cleavage_sites_path",required=True)
    parser.add_argument("-s", "--saved_root",
                        help="saved_root",required=True)
    args = vars(parser.parse_args())
    transcript_gro_sites_path = args['tg']
    transcript_cleavage_sites_path = args['tc']
    inner_gro_sites_path = args['ig']
    inner_cleavage_sites_path = args['ic']
    long_dist_gro_sites_path = args['lg']
    long_dist_cleavage_sites_path = args['lc']
    saved_root = args['saved_root']
    safe_merged_gro_sites_path = saved_root+'/safe_merged_gro_sites'+'.tsv'
    safe_merged_cleavage_sites_path = saved_root+'/safe_merged_cleavage_sites.tsv'    
    exists = [os.path.exists(path) for path in [safe_merged_gro_sites_path,safe_merged_cleavage_sites_path]]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        print('Left only most significant signals')
        ###Read file###
        inner_gro_sites = pd.read_csv(inner_gro_sites_path,sep='\t').dropna(subset=['ref_name'])
        inner_cleavage_sites = pd.read_csv(inner_cleavage_sites_path,sep='\t').dropna(subset=['ref_name'])
        long_dist_gro_sites = pd.read_csv(long_dist_gro_sites_path,sep='\t')
        long_dist_cleavage_sites = pd.read_csv(long_dist_cleavage_sites_path,sep='\t')
        orphan_long_dist_gro_sites = long_dist_gro_sites[long_dist_gro_sites['ref_name'].isna()]
        orphan_long_dist_cleavage_sites = long_dist_cleavage_sites[long_dist_cleavage_sites['ref_name'].isna()]
        transcript_gro_sites = pd.read_csv(transcript_gro_sites_path,sep='\t').dropna(subset=['ref_name'])
        transcript_cleavage_sites = pd.read_csv(transcript_cleavage_sites_path,sep='\t').dropna(subset=['ref_name'])
        #Assign valid GRO sites and cleavage sites to gene###
        print("Get consist site of every transcript")
        inner_gro_sites = consist(inner_gro_sites,'ref_name','tag_count',True)
        transcript_gro_sites = consist(transcript_gro_sites,'ref_name','tag_count',True)
        merged_gro_sites = pd.concat([inner_gro_sites,transcript_gro_sites],sort=False)
        merged_gro_sites = unique_site(merged_gro_sites,'ref_name','evidence_5_end')
        merged_gro_sites = pd.concat([merged_gro_sites,orphan_long_dist_gro_sites],sort=False)
        merged_gro_sites = consist(merged_gro_sites,'ref_name','tag_count',True)
        consist_gro_site = merged_gro_sites[merged_gro_sites['gro_source'].isin(['inner'])]
        
        inner_cleavage_sites = consist(inner_cleavage_sites,'ref_name','read_count',True)
        transcript_cleavage_sites = consist(transcript_cleavage_sites,'ref_name','read_count',True)
        merged_cleavage_sites = pd.concat([inner_cleavage_sites,transcript_cleavage_sites],sort=False)
        merged_cleavage_sites = unique_site(merged_cleavage_sites,'ref_name','evidence_3_end')
        merged_cleavage_sites = pd.concat([merged_cleavage_sites,orphan_long_dist_cleavage_sites],sort=False)
        merged_cleavage_sites = consist(merged_cleavage_sites,'ref_name','read_count',True)
        consist_cleavage_sites = merged_cleavage_sites[merged_cleavage_sites['cleavage_source'].isin(['inner'])]
        print('Export data')
        ###Write data###
        consist_gro_site.to_csv(safe_merged_gro_sites_path,sep='\t',index=False)
        consist_cleavage_sites.to_csv(safe_merged_cleavage_sites_path,sep='\t',index=False)
