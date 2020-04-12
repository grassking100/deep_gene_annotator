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
    parser.add_argument("--drs_path", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    tss_gff_path = os.path.join(args.output_root, 'tss.gff3')
    cleavage_site_gff_path = os.path.join(args.output_root,
                                          'cleavage_site.gff3')
    #tss_path = os.path.join(args.output_root,'tss.tsv')
    #cleavage_site_path = os.path.join(args.output_root,'cleavage_site.tsv')
    paths = [tss_gff_path, cleavage_site_gff_path]
    exists = [os.path.exists(path) for path in paths]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        gro_1 = pd.read_csv(args.gro_1_path, comment='#', sep='\t')
        gro_2 = pd.read_csv(args.gro_2_path, comment='#', sep='\t')
        cleavage_site = pd.read_csv(args.drs_path)
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
        gro = gro.drop('start', 1)
        gro = gro.drop('end', 1)
        gro = gro.assign(
            id=gro.astype(str).apply(lambda x: '_'.join(x), axis=1))
        ###Process cleavage sites data###
        ca_site = cleavage_site[[
            'Chromosome', 'Strand', 'Position', 'Raw DRS read count'
        ]].copy(deep=True)
        ca_site.loc[ca_site['Strand'] == 'fwd', 'Strand'] = '+'
        ca_site.loc[ca_site['Strand'] == 'rev', 'Strand'] = '-'
        ca_site.columns = [
            'chr', 'strand', 'evidence_3_end', 'experimental_score'
        ]
        ca_site.loc[:, 'chr'] = ca_site['chr'].str.replace('chr', '')
        ca_site.loc[ca_site['strand'] == '+', 'evidence_3_end'] += 1
        ca_site.loc[ca_site['strand'] == '-', 'evidence_3_end'] -= 1
        ca_site = ca_site.assign(
            id=ca_site.astype(str).apply(lambda x: '_'.join(x), axis=1))
        ###Drop duplicated ###
        gro = gro.drop_duplicates()
        ca_site = ca_site.drop_duplicates()
        ###Write data##
        #gro.to_csv(tss_path,sep='\t',index=None)
        #ca_site.to_csv(cleavage_site_path,sep='\t',index=None)

        gro['source'] = 'Experiment'
        gro['feature'] = 'GRO site'
        gro['start'] = gro['end'] = gro['evidence_5_end']
        gro['score'] = gro['frame'] = '.'
        gro = gro.drop('evidence_5_end', 1)
        gro = get_gff_with_updated_attribute(gro)
        #gro['attribute'] = 'tag_count=' + gro['tag_count'].astype(str)

        ca_site['source'] = 'Experiment'
        ca_site['feature'] = 'DRS PAC'

        ca_site['start'] = ca_site['end'] = ca_site['evidence_3_end']
        ca_site['score'] = ca_site['frame'] = '.'
        ca_site = ca_site.drop('evidence_3_end', 1)
        ca_site = get_gff_with_updated_attribute(ca_site)
        #ca_site['attribute'] = 'read_count=' + ca_site['read_count'].astype(str)

        write_gff(gro, tss_gff_path)
        write_gff(ca_site, cleavage_site_gff_path)
