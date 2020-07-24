import os, sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_gff,read_gff,InvalidStrandType,write_bed
from sequence_annotation.utils.utils import get_gff_with_updated_attribute,get_gff_with_attribute
from sequence_annotation.utils.get_UTR import get_UTR
from sequence_annotation.preprocess.classify_site import belong_by_boundary
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--official_bed_path",help="Path of selected official gene info file",required=True)
    parser.add_argument("-t", "--tss_path",help="Path of selected TSSs file",required=True)
    parser.add_argument("-c", "--cleavage_site_path",help="Path of cleavage sites file",required=True)
    parser.add_argument("-o", "--output_path",help="Path to save",required=True)
    args = parser.parse_args()
    
    official_bed = read_bed(args.official_bed_path)
    
    UTR = get_UTR(official_bed)
    write_bed(UTR,'UTR.bed')
    tss = get_gff_with_attribute(read_gff(args.tss_path))
    cleavage_site = get_gff_with_attribute(read_gff(args.cleavage_site_path))
    
    transcript_tss = belong_by_boundary(tss,official_bed,'start','start','end','id')
    five_external_utr_tss = belong_by_boundary(tss,UTR[UTR['id']=='five_external_utr'].copy(),'start','start','end','id')
    five_internal_utr_tss = belong_by_boundary(tss,UTR[UTR['id']=='five_internal_utr'].copy(),'start','start','end','id')
    
    transcript_cs = belong_by_boundary(cleavage_site,official_bed,'start','start','end','id')
    three_external_utr_cs = belong_by_boundary(cleavage_site,UTR[UTR['id']=='three_external_utr'].copy(),'start','start','end','id')
    three_internal_utr_cs = belong_by_boundary(cleavage_site,UTR[UTR['id']=='three_internal_utr'].copy(),'start','start','end','id')
    
    five_external_utr_tss['feature'] = 'external_5_UTR_TSS'
    five_internal_utr_tss['feature'] = 'internal_5_UTR_TSS'
    transcript_tss['feature'] = 'transcript_TSS'
    
    three_external_utr_cs['feature'] = 'external_3_UTR_CS'
    three_internal_utr_cs['feature'] = 'internal_3_UTR_CS'
    transcript_cs['feature'] = 'transcript_CS'
    
    utr_tss = pd.concat([five_external_utr_tss,five_internal_utr_tss]).reset_index(drop=True)
    utr_cs = pd.concat([three_external_utr_cs,three_internal_utr_cs]).reset_index(drop=True)
    
    transcript_tss['coord'] = transcript_tss['chr'].astype(str)
    transcript_cs['coord'] = transcript_cs['chr'].astype(str)
    utr_tss['coord'] = utr_tss['chr'].astype(str)
    utr_cs['coord'] = utr_cs['chr'].astype(str)
    for name in ['strand','start','end']:
        transcript_tss['coord'] = transcript_tss['coord']+'_'+transcript_tss[name].astype(str)
        transcript_cs['coord'] = transcript_cs['coord']+'_'+transcript_cs[name].astype(str)
        utr_tss['coord'] = utr_tss['coord']+'_'+utr_tss[name].astype(str)
        utr_cs['coord'] = utr_cs['coord']+'_'+utr_cs[name].astype(str)
    
    other_tss = transcript_tss[~transcript_tss['coord'].isin(utr_tss['coord'])].copy()
    other_cs = transcript_cs[~transcript_cs['coord'].isin(utr_cs['coord'])].copy()
    
    other_tss['feature'] = 'other_TSS'
    other_cs['feature'] = 'other_CS'
    
    all_data = pd.concat([utr_tss,other_tss,utr_cs,other_cs]).reset_index(drop=True)
    
    del all_data['coord']

    all_data = get_gff_with_updated_attribute(all_data)
    write_gff(all_data,args.output_path)
