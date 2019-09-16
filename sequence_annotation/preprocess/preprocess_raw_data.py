import os, sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
import csv
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed, write_bed, write_gff

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("--bed_path",required=True)
    parser.add_argument("--gro_1_path",required=True)
    parser.add_argument("--gro_2_path",required=True)
    parser.add_argument("--cs_path",required=True)
    parser.add_argument("--output_root",required=True)
    #parser.add_argument("--quality_threhold",type=int)
    args = parser.parse_args()

    official_bed_path = os.path.join(args.output_root,'valid_official.bed')
    valid_gro_gff_path = os.path.join(args.output_root,'valid_gro.gff')
    valid_cleavage_site_gff_path = os.path.join(args.output_root,'valid_cleavage_site.gff')
    valid_gro_path = os.path.join(args.output_root,'valid_gro.tsv')
    valid_cleavage_site_path = os.path.join(args.output_root,'valid_cleavage_site.tsv')
    paths = [valid_gro_path,valid_cleavage_site_path,official_bed_path,
             valid_gro_gff_path,valid_cleavage_site_gff_path]
    exists = [os.path.exists(path) for path in paths]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        official_bed = read_bed(args.bed_path)
        valid_chrs = [str(chr_) for chr_ in range(1,6)]
        official_bed = official_bed[official_bed['chr'].isin(valid_chrs)]
        #if args.quality_threhold is not None:
        #    official_bed = official_bed[official_bed['score'].astype(int) >= args.quality_threhold]

        gro_1 = pd.read_csv(args.gro_1_path,comment ='#',sep='\t')
        gro_2 = pd.read_csv(args.gro_2_path,comment ='#',sep='\t')
        cleavage_site = pd.read_csv(args.cs_path)
        ###Process GRO sites data###
        gro_columns = ['chr','strand','Normalized Tag Count','start','end']
        gro_1 = gro_1[gro_columns]
        gro_2 = gro_2[gro_columns]
        gro = gro_1.merge(gro_2,left_on=gro_columns,right_on=gro_columns)
        valid_gro = gro[gro['chr'].isin(valid_chrs)]
        valid_gro.columns = ['chr','strand','tag_count','start','end']
        evidence_5_end = round((valid_gro['end']+ valid_gro['start'])/2)
        valid_gro = valid_gro.assign(evidence_5_end=pd.Series(evidence_5_end).values)
        valid_gro = valid_gro.drop('start', 1)
        valid_gro = valid_gro.drop('end', 1)
        valid_gro = valid_gro.assign(id=valid_gro.astype(str).apply(lambda x: '_'.join(x), axis=1))
        ###Process cleavage sites data###
        ca_site = cleavage_site[['Chromosome','Strand','Position','Raw DRS read count']]
        ca_site.loc[ca_site['Strand']=='fwd','Strand'] = '+'
        ca_site.loc[ca_site['Strand']=='rev','Strand'] = '-'
        ca_site.columns = ['chr','strand','evidence_3_end','read_count']
        ca_site.loc[:,'chr'] = ca_site['chr'].str.replace('chr','')
        valid_ca_site = ca_site[ca_site['chr'].isin(valid_chrs)]
        valid_ca_site = valid_ca_site.assign(id=valid_ca_site.astype(str).apply(lambda x: '_'.join(x), axis=1))
        ###Drop duplicated ###
        valid_gro = valid_gro.drop_duplicates()
        valid_ca_site = valid_ca_site.drop_duplicates()
        ###Write data##
        write_bed(official_bed,official_bed_path)

        valid_gro.to_csv(valid_gro_path,sep='\t',index=None)
        valid_ca_site.to_csv(valid_cleavage_site_path,sep='\t',index=None)
        
        valid_gro['source'] = 'Araport11'
        valid_gro['feature'] = 'GRO site'
        valid_gro['start'] = valid_gro['end'] = valid_gro['evidence_5_end']
        valid_gro['score'] = valid_gro['frame'] = '.'
        valid_gro['attribute'] = 'tag_count=' + valid_gro['tag_count'].astype(str)
        
        valid_ca_site['source'] = 'Araport11'
        valid_ca_site['feature'] = 'cleavage site'
        valid_ca_site['start'] = valid_ca_site['end'] = valid_ca_site['evidence_3_end']
        valid_ca_site['score'] = valid_ca_site['frame'] = '.'   
        valid_ca_site['attribute'] = 'read_count=' + valid_ca_site['read_count'].astype(str)
        
        write_gff(valid_gro,valid_gro_gff_path)
        write_gff(valid_ca_site,valid_cleavage_site_gff_path)
