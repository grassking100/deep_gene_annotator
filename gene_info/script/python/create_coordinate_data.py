import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import consist, coordinate_consist_filter
import os
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root",
                        help="saved_root",required=True)
    parser.add_argument("-g", "--safe_merged_gro_sites_path",
                        help="safe_merged_gro_sites_pathe",required=True)
    parser.add_argument("-c", "--safe_merged_cleavage_sites_path",
                        help="safe_merged_cleavage_sites_path",required=True)
    args = vars(parser.parse_args())
    saved_root = args['saved_root']
    safe_merged_gro_sites_path = args['safe_merged_gro_sites_path']
    safe_merged_cleavage_sites_path = args['safe_merged_cleavage_sites_path']
    clean_merged_data_path = saved_root+'/clean_merged_data.tsv'
    consist_data_path = saved_root+'/coordinate_consist.tsv'
    if os.path.exists(consist_data_path):
        print("Result files are already exist,procedure will be skipped.")
    else:
        safe_merged_gro_sites = pd.read_csv(safe_merged_gro_sites_path,sep='\t')
        safe_merged_cleavage_sites = pd.read_csv(safe_merged_cleavage_sites_path,sep='\t')
        merged_data = safe_merged_cleavage_sites.merge(safe_merged_gro_sites,
                                                       left_on=['chr','strand','ref_name','gene_id'],
                                                       right_on=['chr','strand','ref_name','gene_id'])
        merged_data = merged_data[~ merged_data.duplicated()]
        clean_merged_data = merged_data[~merged_data['evidence_5_end'].isna() & ~merged_data['evidence_3_end'].isna()]
        clean_merged_data['coordinate_start'] =  clean_merged_data[['evidence_5_end','evidence_3_end']].min(1)
        clean_merged_data['coordinate_end'] =  clean_merged_data[['evidence_5_end','evidence_3_end']].max(1)
        clean_merged_data.to_csv(clean_merged_data_path,index=None,sep='\t')
        print('Consist data with gene id')
        consist_data = consist(clean_merged_data,'gene_id','tag_count',False)
        consist_data = consist(consist_data,'gene_id','read_count',False)
        consist_data = coordinate_consist_filter(consist_data,'gene_id','coordinate_start')
        consist_data = coordinate_consist_filter(consist_data,'gene_id','coordinate_end')
        consist_data = consist_data[['ref_name','coordinate_start','coordinate_end']]
        consist_data.to_csv(consist_data_path,index=None,sep='\t')
