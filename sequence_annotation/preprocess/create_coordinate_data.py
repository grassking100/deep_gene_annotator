import os, sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_gff,get_gff_with_updated_attribute,get_gff_with_attribute,write_gff,write_json
from sequence_annotation.preprocess.consist_site import consist
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict

def coordinate_consist_filter(data,group_by,site_name):
    returned = []
    ref_names = set(data[group_by])
    sectors = data.groupby(group_by)
    for name in ref_names:
        sector = sectors.get_group(name)
        value = set(list(sector[site_name]))
        if len(value) == 1:
            returned += sector.to_dict('record')
    return pd.DataFrame.from_dict(returned)

def _read(path):
    data = get_gff_with_attribute(read_gff(path))
    data = data.drop('frame',1)
    data = data.drop('attribute',1)
    data = data.drop('end',1)
    data = data.drop('coord_id',1)
    data = data.drop('coord_ref_id',1)
    return data
        
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will output RNA data id, start site and end site "+
                            "based on TSS and cleavage site data")
    parser.add_argument("-g", "--tss_path",required=True)
    parser.add_argument("-c", "--cleavage_site_path",required=True)
    parser.add_argument("-t","--id_table_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("--stats_path",type=str)
    parser.add_argument("--single_start_end",help="If it is selected, then only RNA data "+
                        "which have start sites and end sites with strongest signal in same gene"+
                        "will be saved",action='store_true')

    args = parser.parse_args()
    
    id_convert_dict = get_id_convert_dict(args.id_table_path)

    if os.path.exists(args.output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        tss = _read(args.tss_path)
        cleavage_site = _read(args.cleavage_site_path)
        tss = tss.rename(columns={'experimental_score':'tss_score','start':'evidence_5_end',
                                  'source':'tss_source','feature':'tss_feature'})
        cleavage_site = cleavage_site.rename(columns={'experimental_score':'cleavage_site_score',
                                                      'start':'evidence_3_end',
                                                      'source':'cleavage_site_source',
                                                      'feature':'cleavage_site_feature'})
        merged_data = cleavage_site.merge(tss,left_on=['chr','strand','ref_name'],right_on=['chr','strand','ref_name'])
        number = {}
        cleaned = merged_data[(~merged_data['evidence_5_end'].isna()) & (~merged_data['evidence_3_end'].isna())]
        if set(cleaned['strand']) != set(['+','-']):
            raise Exception("Invalid strand")
        
        number['transcript'] = len(cleaned)
        print("Merged boundary number is {}".format(len(cleaned)))
        plus_index = cleaned['strand'] == '+'
        minus_index = cleaned['strand'] == '-'
        plus_valid_order = (cleaned.loc[plus_index,'evidence_5_end'] <= cleaned.loc[plus_index,'evidence_3_end']).index
        minus_valid_order = (cleaned.loc[minus_index,'evidence_5_end'] >= cleaned.loc[minus_index,'evidence_3_end']).index
        cleaned = cleaned.loc[list(plus_valid_order)+list(minus_valid_order),:]
        number['valid transcript'] = len(cleaned)
        print("Merged valid boundary number is {}".format(len(cleaned)))
        evidence_site = cleaned[['evidence_5_end','evidence_3_end']]
        cleaned['start'] =  evidence_site.min(1)
        cleaned['end'] =  evidence_site.max(1)
        print('Consist data with gene id')
        cleaned['gene_id'] = [id_convert_dict[id_] for id_ in list(cleaned['ref_name'])]

        if args.single_start_end:
            cleaned = consist(cleaned,'gene_id','tss_score',drop_duplicated=False)
            cleaned = consist(cleaned,'gene_id','cleavage_site_score',drop_duplicated=False)
            cleaned = coordinate_consist_filter(cleaned,'gene_id','start')
            cleaned = coordinate_consist_filter(cleaned,'gene_id','end')
            number['transcript which its gene has single start and single end'] = len(cleaned)
        cleaned = get_gff_with_updated_attribute(cleaned)
        cleaned['source'] = 'Experiment'
        cleaned['feature'] = 'boundary'
        cleaned['score'] = cleaned['frame'] = '.'
        write_gff(cleaned,args.output_path)

        if args.stats_path is not None:
            write_json(number,args.stats_path)
