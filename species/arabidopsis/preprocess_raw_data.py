import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import write_gff, get_gff_with_updated_attribute,write_json,create_folder

def _get_ids(gff):
    ids = list(gff['chr']+"_"+gff['start'].astype(str)+"_"+gff['strand'])
    return ids

def _is_duplicated(gff):
    ids = _get_ids(gff)
    return len(ids)!=len(set(ids))

def _get_unique_id_len(data):
    return len(set(_get_ids(data)))

def _convert_gro_to_gff(gro):
    gro_columns = ['chr', 'strand', 'Normalized Tag Count', 'start', 'end']
    gro = gro[gro_columns]
    gro.columns = ['chr', 'strand', 'experimental_score', 'start', 'end']
    evidence_5_end = round((gro['end'] + gro['start']) / 2)
    gro = gro.assign(evidence_5_end=pd.Series(evidence_5_end).values)
    gro['source'] = 'Experiment'
    gro['feature'] = 'GRO site'
    gro['start'] = gro['end'] = gro['evidence_5_end']
    gro['frame'] = gro['score'] = '.'
    gro = gro[gro['chr'].isin(['1','2','3','4','5'])]
    gro = gro.drop('evidence_5_end', 1)
    gro = get_gff_with_updated_attribute(gro)
    return gro

def _convert_pac_to_gff(pac):
    pac_site = pac[['chr', 'strand', 'coord','tot_tag']].copy(deep=True)
    pac_site.columns = ['chr', 'strand', 'evidence_3_end', 'experimental_score']
    pac_site['source'] = 'Experiment'
    pac_site['feature'] = 'PAT-Seq PAC'
    pac_site['start'] = pac_site['end'] = pac_site['evidence_3_end']
    pac_site['frame'] = pac_site['score'] = '.'
    pac_site = pac_site[pac_site['chr'].isin(['1','2','3','4','5'])]
    pac_site = pac_site.drop('evidence_3_end', 1)
    pac_site = get_gff_with_updated_attribute(pac_site)
    return pac_site

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("--gro_1_path", required=True)
    parser.add_argument("--gro_2_path", required=True)
    parser.add_argument("--pac_path", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    tss_gff_path = os.path.join(args.output_root, 'tss.gff3')
    cs_gff_path = os.path.join(args.output_root,'cleavage_site.gff3')
    preprocess_stats_path = os.path.join(args.output_root,'preprocess_stats.json')
    paths = [tss_gff_path, cs_gff_path]
    exists = [os.path.exists(path) for path in paths]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        create_folder(args.output_root)
        gro_1 = pd.read_csv(args.gro_1_path, comment='#', sep='\t')
        gro_2 = pd.read_csv(args.gro_2_path, comment='#', sep='\t')
        cs = pd.read_csv(args.pac_path,dtype={'chr':str})
        ###Process GRO sites data###
        #The Normalized Tag Count at both data on the same location would be same, because they are from GRO dataset
        gro_1 = _convert_gro_to_gff(gro_1)
        gro_2 = _convert_gro_to_gff(gro_2)
        gro_1_num = _get_unique_id_len(gro_1)
        gro_2_num = _get_unique_id_len(gro_2)
        gro = gro_1.merge(gro_2)
        ###Process cleavage sites data###
        raw_pac = _convert_pac_to_gff(cs)
        raw_pac_num = _get_unique_id_len(raw_pac)
        ###Drop duplicated ###
        gro = gro.drop_duplicates()
        pac = raw_pac.drop_duplicates()
        gro_num = _get_unique_id_len(gro)
        pac_num = _get_unique_id_len(pac)
        ###Write data##
        if _is_duplicated(gro) or _is_duplicated(pac):
            raise Exception()
        
        write_gff(gro, tss_gff_path)
        write_gff(pac, cs_gff_path)
        stats = {
            'Raw GRO dataset 1 number':gro_1_num,
            'Raw GRO dataset 2 number':gro_2_num,
            'Consistent GRO dataset number':gro_num,
            'Raw PAC dataset number':raw_pac_num,
            'PAC dataset number':pac_num
        }
        write_json(stats,preprocess_stats_path)
