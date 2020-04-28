import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed, write_gff, get_gff_with_updated_attribute

def preprocess_tss_data(path):
    tss_bed = read_bed(path)
    gff = bed2gff(tss_bed)
    gff['experimental_score'] = gff['score']
    gff['frame'] = '.'
    gff['attribute'] = '.'
    gff['source'] = '.'
    gff['feature'] = 'TSS'
    gff = get_gff_with_updated_attribute(gff)
    gff = gff.drop_duplicates()
    return gff

def preprocess_cs_data(path):
    df = pd.read_csv(path,sep='\t')
    df = df.assign(experimental_score=df.iloc[:,7:].max(1))
    df = df[['chrom','strand','start','end','score']]
    gff = df.rename(columns={'chrom':'chr'})
    gff['frame'] = '.'
    gff['attribute'] = '.'
    gff['source'] = '.'
    gff['feature'] = 'Cleavage site'
    gff = get_gff_with_updated_attribute(gff)
    gff = gff.drop_duplicates()
    return gff

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("--tss_bed_path", required=True)
    parser.add_argument("--cleavage_site_tsv_path", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    tss_gff_path = os.path.join(args.output_root, 'tss.gff3')
    cs_gff_path = os.path.join(args.output_root,'cleavage_site.gff3')
    paths = [tss_gff_path, cs_gff_path]
    exists = [os.path.exists(path) for path in paths]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        valid_chroms = ['chrX','chr2L','chr2R','chr3L','chr3R','chr4']
        ###Process TSS data###
        tss_gff = preprocess_tss_data(args.tss_bed_path)
        cs_gff = preprocess_cs_data(args.cleavage_site_tsv_path)
        tss_gff = tss_gff[tss_gff['chr'].isin(valid_chroms)]
        cs_gff = cs_gff[cs_gff['chr'].isin(valid_chroms)]
        write_gff(tss_gff, tss_gff_path)
        write_gff(cs_gff, cs_gff_path)
