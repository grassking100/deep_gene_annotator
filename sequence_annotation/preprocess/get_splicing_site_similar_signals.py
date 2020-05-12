import sys,os
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_gff, read_fasta,write_bed,create_folder,get_gff_with_attribute,find_substr
from sequence_annotation.preprocess.utils import INTRON_TYPES, RNA_TYPES
from sequence_annotation.preprocess.gff2bed import simple_gff2bed
from sequence_annotation.preprocess.signal_analysis import get_donor_site_region,get_acceptor_site_region

def set_coord_id(bed):
    bed['coord_id'] = bed['chr'] + "_" + bed['strand'] + "_" + bed['start'].astype(str) + "_" + bed['end'].astype(str)
    
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
    donor_bed = get_donor_site_region(gff, dist, dist)
    acceptor_bed = get_acceptor_site_region(gff, dist, dist)
    transcript_group = transcript_bed.groupby('id')
    
    donor_group = donor_bed.groupby('transcript_source')
    acceptor_group = acceptor_bed.groupby('transcript_source')
    potential_donor_bed = []
    potential_acceptor_bed = []
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
            potential_donor_bed.append(item)
        for index in potential_acceptor_indice:
            item = dict(template)
            item.update({'start':index-dist,'end':index+dist})
            potential_acceptor_bed.append(item)

    potential_donor_bed = pd.DataFrame.from_dict(potential_donor_bed)
    potential_acceptor_bed = pd.DataFrame.from_dict(potential_acceptor_bed)
    #Get splicing site like motif
    set_coord_id(donor_bed)
    set_coord_id(acceptor_bed)
    set_coord_id(potential_donor_bed)
    set_coord_id(potential_acceptor_bed)
    donor_bed = donor_bed[donor_bed['start']>0]
    acceptor_bed = acceptor_bed[acceptor_bed['start']>0]
    potential_donor_bed = potential_donor_bed[potential_donor_bed['start']>0]
    potential_acceptor_bed = potential_acceptor_bed[potential_acceptor_bed['start']>0]

    fake_donor_bed = potential_donor_bed[~potential_donor_bed['coord_id'].isin(donor_bed['coord_id'])]
    fake_acceptor_bed = potential_acceptor_bed[~potential_acceptor_bed['coord_id'].isin(acceptor_bed['coord_id'])]
    #Get fasta
    fake_donor_bed_path = os.path.join(output_root,'fake_donor.bed')
    fake_donor_fasta_path = os.path.join(output_root,'fake_donor.fasta')
    write_bed(fake_donor_bed,fake_donor_bed_path)
    os.system("bedtools getfasta -s -fi {} -bed {} -fo {}".format(fasta_path,fake_donor_bed_path,fake_donor_fasta_path))

    fake_acceptor_bed_path = os.path.join(output_root,'fake_acceptor.bed')
    fake_acceptor_fasta_path = os.path.join(output_root,'fake_acceptor.fasta')
    write_bed(fake_acceptor_bed,fake_acceptor_bed_path)
    os.system("bedtools getfasta -s -fi {} -bed {} -fo {}".format(fasta_path,fake_acceptor_bed_path,fake_acceptor_fasta_path))

    donor_bed_path = os.path.join(output_root,'donor.bed')
    donor_fasta_path = os.path.join(output_root,'donor.fasta')
    write_bed(donor_bed,donor_bed_path)
    os.system("bedtools getfasta -s -fi {} -bed {} -fo {}".format(fasta_path,donor_bed_path,donor_fasta_path))

    acceptor_bed_path = os.path.join(output_root,'acceptor.bed')
    acceptor_fasta_path = os.path.join(output_root,'acceptor.fasta')
    write_bed(acceptor_bed,acceptor_bed_path)
    os.system("bedtools getfasta -s -fi {} -bed {} -fo {}".format(fasta_path,acceptor_bed_path,acceptor_fasta_path))


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
    
