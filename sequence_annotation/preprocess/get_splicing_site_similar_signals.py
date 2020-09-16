import sys,os
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder,find_substr
from sequence_annotation.file_process.utils import read_gff, read_fasta,write_bed,get_gff_with_attribute, RNA_TYPES
from sequence_annotation.file_process.gff2bed import simple_gff2bed
from sequence_annotation.preprocess.signal_analysis import get_donor_site_region,get_acceptor_site_region

def get_bed_wtih_site_id(bed,is_donor=True):
    plus_bed = bed[bed['strand']=='+'].copy()
    minus_bed = bed[bed['strand']=='-'].copy()
    plus_bed['coord'] = minus_bed['coord'] = None
    plus_inron=plus_bed['start'].astype(str)
    minus_inron=minus_bed['start'].astype(str)
    
    if is_donor:
        plus_exon=(plus_bed['start']-1).astype(str)
        minus_exon=(minus_bed['start']+1).astype(str)
        plus_bed['coord'] = plus_bed['chr'] + "_" + plus_bed['strand'] + "_" + plus_exon+"_"+plus_inron
        minus_bed['coord'] = bed['chr'] + "_" + bed['strand'] + "_" + minus_inron+"_"+minus_exon
    else:
        plus_exon=(plus_bed['start']+1).astype(str)
        minus_exon=(minus_bed['start']-1).astype(str)
        plus_bed['coord'] = bed['chr'] + "_" + bed['strand'] + "_" + plus_inron+"_"+plus_exon
        minus_bed['coord'] = bed['chr'] + "_" + bed['strand'] + "_" + minus_exon+"_"+minus_inron

    bed = pd.concat([plus_bed,minus_bed])
    return bed
    
def get_splicing_site_motifs(gff_path,fasta_path,dist,output_root):
    #Get transcript fasta
    create_folder(output_root)
    gff = read_gff(gff_path)
    gff = get_gff_with_attribute(gff)
    transcript_gff = gff[gff['feature'].isin(RNA_TYPES)]
    transcript_bed = simple_gff2bed(transcript_gff)
    transcript_bed_path = os.path.join(output_root,'transcript.bed')
    transcript_fasta_path = os.path.join(output_root,'transcript.fasta')
    write_bed(transcript_bed,transcript_bed_path)
    os.system("bedtools getfasta -s -name -fi {} -bed {} -fo {}".format(fasta_path,transcript_bed_path,transcript_fasta_path))
    fasta = read_fasta(transcript_fasta_path)
    #Get splicing site motif
    #print(gff.head()['parent'])
    donor = get_donor_site_region(gff, dist, dist)
    acceptor = get_acceptor_site_region(gff, dist, dist)
    transcript_group = transcript_bed.groupby('id')
    
    potential_donor = []
    potential_acceptor = []
    for region_id, seq in fasta.items():
        #Get splicing site potential motif
        strand = list(transcript_group.get_group(region_id)['strand'])[0]
        chrom = list(transcript_group.get_group(region_id)['chr'])[0]
        potential_donor_indice = find_substr("GT", seq)
        potential_acceptor_indice = find_substr("AG", seq, shift_value=1)
        #print(potential_donor_indice)
        if strand == '+':
            start = list(transcript_group.get_group(region_id)['start'])[0]
            potential_donor_indice = [start+indice for indice in potential_donor_indice]
            potential_acceptor_indice = [start+indice for indice in potential_acceptor_indice]
        else:
            end = list(transcript_group.get_group(region_id)['end'])[0]
            potential_donor_indice = [end-indice for indice in potential_donor_indice]
            potential_acceptor_indice = [end-indice for indice in potential_acceptor_indice]
            
        template = {'chr':chrom,'strand':strand,'id':'.','score':'.'}
        for index in potential_donor_indice:
            item = dict(template)
            item.update({'start':index-dist,'end':index+dist})
            potential_donor.append(item)
        for index in potential_acceptor_indice:
            item = dict(template)
            item.update({'start':index-dist,'end':index+dist})
            potential_acceptor.append(item)

    potential_donor = pd.DataFrame.from_dict(potential_donor)
    potential_acceptor = pd.DataFrame.from_dict(potential_acceptor)
    #Get splicing site like motif
    donor = get_bed_wtih_site_id(donor)
    acceptor = get_bed_wtih_site_id(acceptor,is_donor=False)
    potential_donor = get_bed_wtih_site_id(potential_donor)
    potential_acceptor = get_bed_wtih_site_id(potential_acceptor,is_donor=False)

    donor = donor[donor['start']>0]
    acceptor = acceptor[acceptor['start']>0]
    potential_donor = potential_donor[potential_donor['start']>0]
    potential_acceptor = potential_acceptor[potential_acceptor['start']>0]

    real_coord = set(list(donor['coord'])+list(acceptor['coord']))
    fake_donor = potential_donor[~potential_donor['coord'].isin(real_coord)]
    fake_acceptor = potential_acceptor[~potential_acceptor['coord'].isin(real_coord)]
    
    #Get fasta
    fake_donor_path = os.path.join(output_root,'fake_donor.bed')
    fake_donor_fasta_path = os.path.join(output_root,'fake_donor.fasta')
    write_bed(fake_donor,fake_donor_path)
    os.system("bedtools getfasta -s -fi {} -bed {} -fo {}".format(fasta_path,fake_donor_path,fake_donor_fasta_path))

    fake_acceptor_path = os.path.join(output_root,'fake_acceptor.bed')
    fake_acceptor_fasta_path = os.path.join(output_root,'fake_acceptor.fasta')
    write_bed(fake_acceptor,fake_acceptor_path)
    os.system("bedtools getfasta -s -fi {} -bed {} -fo {}".format(fasta_path,fake_acceptor_path,fake_acceptor_fasta_path))

    donor_path = os.path.join(output_root,'donor.bed')
    donor_fasta_path = os.path.join(output_root,'donor.fasta')
    write_bed(donor,donor_path)
    os.system("bedtools getfasta -s -fi {} -bed {} -fo {}".format(fasta_path,donor_path,donor_fasta_path))

    acceptor_path = os.path.join(output_root,'acceptor.bed')
    acceptor_fasta_path = os.path.join(output_root,'acceptor.fasta')
    write_bed(acceptor,acceptor_path)
    os.system("bedtools getfasta -s -fi {} -bed {} -fo {}".format(fasta_path,acceptor_path,acceptor_fasta_path))


if __name__ == '__main__':
    parser = ArgumentParser(
        description="This program will create splicing site related motifs")
    parser.add_argument("-i","--gff_path",required=True,help="Path of GFF file")
    parser.add_argument("-f","--fasta_path",required=True,help="Path of fasta file")
    parser.add_argument("-o","--output_root",required=True,
                        help="Path of output gff file")
    parser.add_argument("-r","--radius",required=True,type=int)
    args = parser.parse_args()
    get_splicing_site_motifs(args.gff_path,args.fasta_path,
                             args.radius,args.output_root)
    
