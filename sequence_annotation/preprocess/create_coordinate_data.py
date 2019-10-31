import os
import pandas as pd
from argparse import ArgumentParser
from utils import consist, coordinate_consist_filter, get_id_table

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will output RNA data id, start site and end site "+
                            "based on GRO and DRS site data")
    parser.add_argument("-g", "--safe_merged_gro_sites_path",required=True)
    parser.add_argument("-c", "--safe_merged_cleavage_sites_path",required=True)
    parser.add_argument("-t","--id_convert_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("--single_start_end",help="If it is selected, then only RNA data "+
                        "which have start sites and end sites with strongest signal in same gene"+
                        "will be saved",action='store_true')

    args = parser.parse_args()
    id_convert = get_id_table(args.id_convert_path)
    if os.path.exists(args.output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        safe_merged_gro_sites = pd.read_csv(args.safe_merged_gro_sites_path,sep='\t')
        safe_merged_cleavage_sites = pd.read_csv(args.safe_merged_cleavage_sites_path,sep='\t')
        merged_data = safe_merged_cleavage_sites.merge(safe_merged_gro_sites,
                                                       left_on=['chr','strand','ref_name'],
                                                       right_on=['chr','strand','ref_name'])
        merged_data = merged_data[~merged_data.duplicated()]
        clean_merged_data = merged_data[~merged_data['evidence_5_end'].isna() & ~merged_data['evidence_3_end'].isna()]
        evidence_sites = clean_merged_data[['evidence_5_end','evidence_3_end']]
        clean_merged_data['coordinate_start'] =  evidence_sites.min(1)
        clean_merged_data['coordinate_end'] =  evidence_sites.max(1)
        print('Consist data with gene id')
        clean_merged_data['gene_id'] = [id_convert[id_] for id_ in list(clean_merged_data['ref_name'])]

        if args.single_start_end:
            clean_merged_data = consist(clean_merged_data,'gene_id','tag_count',False)
            clean_merged_data = consist(clean_merged_data,'gene_id','read_count',False)
            clean_merged_data = coordinate_consist_filter(clean_merged_data,'gene_id','coordinate_start')
            clean_merged_data = coordinate_consist_filter(clean_merged_data,'gene_id','coordinate_end')
        consist_data = clean_merged_data[['ref_name','coordinate_start','coordinate_end']]
        consist_data.to_csv(args.output_path,index=None,sep='\t')