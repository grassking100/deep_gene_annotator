import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import consist, get_id_table
import pandas as pd
from argparse import ArgumentParser
def get_gene_names(df,name,convert_table):
    return [convert_table[name] for name in df[name]]
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will used mRNA_bed12 data to create annotated genome\n"+
                            "and it will selecte region from annotated genome according to the\n"+
                            "selected_region and selected region will save to output_path in h5 format")
    parser.add_argument("--dist_gro_sites_path",
                        help="dist_gro_sites_path",required=True)
    parser.add_argument("--dist_cleavage_sites_path",
                        help="dist_cleavage_sites_path",required=True)
    parser.add_argument("--inner_gro_sites_path",
                        help="inner_gro_sites_path",required=True)
    parser.add_argument("--inner_cleavage_sites_path",
                        help="inner_cleavage_sites_path",required=True)
    parser.add_argument("--long_dist_gro_sites_path",
                        help="long_dist_gro_sites_path",required=True)
    parser.add_argument("--long_dist_cleavage_sites_path",
                        help="long_dist_cleavage_sites_path",required=True)
    parser.add_argument("--orf_inner_gro_sites_path",
                        help="orf_inner_gro_sites_path",required=True)
    parser.add_argument("--orf_inner_cleavage_sites_path",
                        help="orf_inner_cleavage_sites_path",required=True)
    parser.add_argument("-s", "--saved_root",
                        help="saved_root",required=True)
    parser.add_argument("--id_convert_path",help="id_convert_path",required=True)
    args = vars(parser.parse_args())
    dist_gro_sites_path = args['dist_gro_sites_path']
    dist_cleavage_sites_path = args['dist_cleavage_sites_path']
    inner_gro_sites_path = args['inner_gro_sites_path']
    inner_cleavage_sites_path = args['inner_cleavage_sites_path']
    long_dist_gro_sites_path = args['long_dist_gro_sites_path']
    long_dist_cleavage_sites_path = args['long_dist_cleavage_sites_path']
    orf_inner_gro_sites_path = args['orf_inner_gro_sites_path']
    orf_inner_cleavage_sites_path = args['orf_inner_cleavage_sites_path']
    saved_root = args['saved_root']
    id_convert_path = args['id_convert_path']
    id_convert = pd.read_csv(id_convert_path,sep='\t',index_col=0).to_dict()['gene_id']
    id_convert = get_id_table(id_convert_path)
    safe_merged_gro_sites_path = saved_root+'/safe_merged_gro_sites'+'.tsv'
    safe_merged_cleavage_sites_path = saved_root+'/safe_merged_cleavage_sites.tsv'
    print('Left only most significant signals')
    exists = [os.path.exists(path) for path in [safe_merged_gro_sites_path,safe_merged_cleavage_sites_path]]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        ###Read file###
        dist_gro_sites = pd.read_csv(dist_gro_sites_path,sep='\t').dropna(subset=['ref_name'])
        dist_cleavage_sites = pd.read_csv(dist_cleavage_sites_path,sep='\t').dropna(subset=['ref_name'])
        inner_gro_sites = pd.read_csv(inner_gro_sites_path,sep='\t').dropna(subset=['ref_name'])
        inner_cleavage_sites = pd.read_csv(inner_cleavage_sites_path,sep='\t').dropna(subset=['ref_name'])
        long_dist_gro_sites = pd.read_csv(long_dist_gro_sites_path,sep='\t').dropna(subset=['ref_name'])
        long_dist_cleavage_sites = pd.read_csv(long_dist_cleavage_sites_path,sep='\t').dropna(subset=['ref_name'])
        orf_inner_gro_sites = pd.read_csv(orf_inner_gro_sites_path,sep='\t').dropna(subset=['ref_name'])
        orf_inner_cleavage_sites = pd.read_csv(orf_inner_cleavage_sites_path,sep='\t').dropna(subset=['ref_name'])
        #Assign valid GRO sites and cleavage sites to gene###
        print("Merge belonging data")
        dist_gro_sites['gro_source'] = 'dist'
        inner_gro_sites['gro_source'] = 'inner'
        long_dist_gro_sites['gro_source'] = 'long_dist'
        orf_inner_gro_sites['gro_source'] = 'orf_inner'
        merged_gro_sites = pd.concat([dist_gro_sites,inner_gro_sites,long_dist_gro_sites])
        
        dist_cleavage_sites['cleavage_source'] = 'dist'
        inner_cleavage_sites['cleavage_source'] = 'inner'
        long_dist_cleavage_sites['cleavage_source'] = 'long_dist'
        orf_inner_cleavage_sites['cleavage_source'] = 'orf_inner'
        
        merged_cleavage_sites = pd.concat([dist_cleavage_sites,inner_cleavage_sites,long_dist_cleavage_sites])
        ##
        gro_names = get_gene_names(merged_gro_sites,'ref_name',id_convert)
        merged_gro_sites = merged_gro_sites.assign(gene_id=pd.Series(gro_names).values)
        cs_names = get_gene_names(merged_cleavage_sites,'ref_name',id_convert)
        merged_cleavage_sites = merged_cleavage_sites.assign(gene_id=pd.Series(cs_names).values)
        ##

        ###Clean data without invalid sites###
        print('Clean and export data')
        long_dist_gro_site_id = set(merged_gro_sites[merged_gro_sites['gro_source'].isin(['long_dist'])]['gene_id'])
        
        safe_merged_gro_site = merged_gro_sites[merged_gro_sites['gro_source'].isin(['dist','inner']) ]
        
        long_dist_cleavage_site_id = set(merged_cleavage_sites[merged_cleavage_sites['cleavage_source'].isin(['long_dist'])]['gene_id'])
        safe_merged_cleavage_sites = merged_cleavage_sites[merged_cleavage_sites['cleavage_source'].isin(['dist','inner'])& merged_cleavage_sites['gene_id'].isin(long_dist_cleavage_site_id)]
        
        
        #safe_merged_gro_site = consist(safe_merged_gro_site,'gene_id','tag_count',True)
        #safe_merged_cleavage_sites = consist(safe_merged_cleavage_sites,'gene_id','read_count',True)
        
        ###Write data###

        safe_merged_gro_site.to_csv(safe_merged_gro_sites_path,sep='\t',index=False)
        safe_merged_cleavage_sites.to_csv(safe_merged_cleavage_sites_path,sep='\t',index=False)
