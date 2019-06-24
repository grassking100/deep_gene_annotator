import os, sys
import pandas as pd
from argparse import ArgumentParser
import csv
from utils import read_bed, write_bed
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("--biomart_path",
                        help="biomart_path",required=True)
    parser.add_argument("--bed_path",
                        help="bed_path",required=True)
    parser.add_argument("--gro_1_path",
                        help="gro_1_path",required=True)
    parser.add_argument("--gro_2_path",
                        help="gro_2_path",required=True)
    parser.add_argument("--cs_path",
                        help="cs_path",required=True)
    parser.add_argument("--saved_root",
                        help="saved_root",required=True)
    args = parser.parse_args()

    valid_official_coding_bed_path = os.path.join(args.saved_root,'valid_official_coding.bed')
    valid_gro_path = os.path.join(args.saved_root,'valid_gro.tsv')
    valid_cleavage_site_path = os.path.join(args.saved_root,'valid_cleavage_site.tsv')
    id_convert_path = os.path.join(args.saved_root,'id_convert.tsv')
    paths = [valid_official_coding_bed_path,
             valid_gro_path,valid_cleavage_site_path]
    exists = [os.path.exists(path) for path in paths]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        ###Read file###
        biomart_gene_info = pd.read_csv(args.biomart_path)
        biomart_gene_info = biomart_gene_info[['Gene stable ID','Transcript stable ID','Transcript type']]
        biomart_gene_info.columns = ['gene_id','transcript_id','transcript_type']
        official_coding_bed = read_bed(args.bed_path)
        gro_1 = pd.read_csv(args.gro_1_path,comment ='#',sep='\t')
        gro_2 = pd.read_csv(args.gro_2_path,comment ='#',sep='\t')
        cleavage_site = pd.read_csv(args.cs_path)
        valid_chrs = [str(chr_) for chr_ in range(1,6)]
        ###Process araport_11_gene_info###
        biomart_coding = biomart_gene_info[biomart_gene_info['transcript_type']=='protein_coding']
        official_coding_bed['chr'] = official_coding_bed['chr'].str.replace('Chr','')
        official_coding_bed = official_coding_bed[official_coding_bed['chr'].isin(valid_chrs)]
        ###Create id_convert table###
        id_convert = {}
        for item in biomart_coding.to_dict('record'):
            id_convert[item['transcript_id']] = item['gene_id']
        ###Creatre valid geen and mRNA id###
        valid_transcript_ids = set(official_coding_bed['id']).intersection(set(biomart_coding['transcript_id']))
        valid_gene_ids = [id_convert[id_] for id_ in valid_transcript_ids]
        ###Create valid_official_araport11_coding###
        valid_official_coding_bed = official_coding_bed[official_coding_bed['id'].isin(valid_transcript_ids)]
        ###Process GRO sites data###
        gro_columns = ['chr','strand','Normalized Tag Count','start','end']
        gro_1 = gro_1[gro_columns]
        gro_2 = gro_2[gro_columns]
        gro = gro_1.merge(gro_2,left_on=gro_columns,right_on=gro_columns)
        valid_gro = gro[gro['chr'].isin(valid_chrs)]
        valid_gro.columns = ['chr','strand','tag_count','start','end']
        evidence_5_end = round((valid_gro['end']+ valid_gro['start'])/2)
        valid_gro = valid_gro.assign(evidence_5_end=pd.Series(evidence_5_end).values)
        ###Process cleavage sites data###
        cleavage_site = cleavage_site[['Chromosome','Strand','Position','Raw DRS read count']]
        cleavage_site.loc[cleavage_site['Strand']=='fwd','Strand'] = '+'
        cleavage_site.loc[cleavage_site['Strand']=='rev','Strand'] = '-'
        cleavage_site.columns = ['chr','strand','evidence_3_end','read_count']
        cleavage_site['chr'] = cleavage_site['chr'].str.replace('chr','')
        valid_cleavage_site = cleavage_site[cleavage_site['chr'].isin(valid_chrs)]
        ###Drop duplicated ###
        valid_gro = valid_gro.drop_duplicates()
        valid_cleavage_site = valid_cleavage_site.drop_duplicates()
        ###Write data##
        write_bed(valid_official_coding_bed,valid_official_coding_bed_path)
        valid_gro = valid_gro.drop('start', 1)
        valid_gro = valid_gro.drop('end', 1)
        valid_gro.to_csv(valid_gro_path,sep='\t',index=None)
        valid_cleavage_site.to_csv(valid_cleavage_site_path,sep='\t',index=None)
        df = pd.DataFrame.from_dict(id_convert,'index')
        df.index.name = 'transcript_id'
        df.columns = ['gene_id']
        df.to_csv(id_convert_path,sep='\t')