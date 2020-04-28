import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import write_gff, get_gff_with_updated_attribute

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
    paths = [tss_gff_path, cs_gff_path]
    exists = [os.path.exists(path) for path in paths]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        gro_1 = pd.read_csv(args.gro_1_path, comment='#', sep='\t')
        gro_2 = pd.read_csv(args.gro_2_path, comment='#', sep='\t')
        cs = pd.read_csv(args.pac_path,dtype={'chr':str})
        ###Process GRO sites data###
        #The Normalized Tag Count at both data on the same location would be same, because they are from GRO dataset
        gro_columns = ['chr', 'strand', 'Normalized Tag Count', 'start', 'end']
        gro_1 = gro_1[gro_columns]
        gro_2 = gro_2[gro_columns]
        gro = gro_1.merge(gro_2, left_on=gro_columns, right_on=gro_columns)
        print("The number of GRO are {} and {},the common one is {}".format(
            len(gro_1), len(gro_2), len(gro)))
        gro.columns = ['chr', 'strand', 'experimental_score', 'start', 'end']
        evidence_5_end = round((gro['end'] + gro['start']) / 2)
        gro = gro.assign(evidence_5_end=pd.Series(evidence_5_end).values)
        ###Process cleavage sites data###
        pac_site = cs[['chr', 'strand', 'coord',
                                  'tot_tag']].copy(deep=True)
        pac_site.columns = [
            'chr', 'strand', 'evidence_3_end', 'experimental_score'
        ]
        ###Drop duplicated ###
        gro = gro.drop_duplicates()
        pac_site = pac_site.drop_duplicates()
        ###Write data##
        gro['source'] = 'Experiment'
        gro['feature'] = 'GRO site'
        gro['start'] = gro['end'] = gro['evidence_5_end']
        gro['frame'] = gro['score'] = '.'
        gro = gro[gro['chr'].isin(['1','2','3','4','5'])]
        gro = gro.drop('evidence_5_end', 1)
        gro = get_gff_with_updated_attribute(gro)

        pac_site['source'] = 'Experiment'
        pac_site['feature'] = 'PAT-Seq PAC'
        pac_site['start'] = pac_site['end'] = pac_site['evidence_3_end']
        pac_site['frame'] = pac_site['score'] = '.'
        pac_site = pac_site[pac_site['chr'].isin(['1','2','3','4','5'])]
        pac_site = pac_site.drop('evidence_3_end', 1)
        pac_site = get_gff_with_updated_attribute(pac_site)

        write_gff(gro, tss_gff_path)
        write_gff(pac_site, cs_gff_path)
