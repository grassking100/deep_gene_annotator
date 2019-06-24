import os, sys
sys.path.append(os.path.dirname(__file__))
from Bio import SeqIO
from utils import read_bed, write_bed, simply_coord
import pandas as pd
from argparse import ArgumentParser

def getfasta(genome_path,bed_path,saved_path,use_split=False):
    if use_split:
        command = 'bedtools getfasta -s -name -split -fi {} -bed {} -fo {}'
    else:    
        command = 'bedtools getfasta -s -name -fi {} -bed {} -fo {}'
    command = command.format(genome_path,bed_path,saved_path)
    print(command)
    os.system(command)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root",help="saved_root",required=True)
    parser.add_argument("-b", "--bed_path",help="bed_path",required=True)
    parser.add_argument("-g", "--genome_path",help="genome_path",required=True)
    parser.add_argument("-p", "--peptide_path",help="peptide_path",required=True)
    parser.add_argument("-t", "--TSS_radius",help="TSS_radius",required=True)
    parser.add_argument("-c", "--cleavage_radius",help="cleavage_radius",required=True)
    parser.add_argument("-d", "--donor_radius",help="donor_radius",required=True)
    parser.add_argument("-a", "--accept_radius",help="accept_radius",required=True)
    args = parser.parse_args()
    root_path = "/".join(sys.argv[0].split('/')[:-1])
    bed_name = ".".join(args.bed_path.split('/')[-1].split(".")[:-1])
    script_root = '{}/../bash'.format(root_path)
    tss_around_path = 'transcription_start_site_with_radius_{}'.format(args.TSS_radius)
    cleavage_around_path = 'cleavage_site_with_radius_{}'.format(args.cleavage_radius)
    donor_around_path = 'splice_donor_site_with_radius_{}'.format(args.donor_radius)
    accept_around_path = 'splice_accept_site_with_radius_{}'.format(args.accept_radius)
    tool_names = ['transcription_start_site','splice_donor_site','splice_accept_site','cleavage_site']
    site_radii = [args.TSS_radius,args.donor_radius,args.accept_radius,args.cleavage_radius]
    for tool_name,radius in zip(tool_names,site_radii):
        command = "bash {}/{}.sh {} {}".format(script_root,tool_name,args.bed_path,radius)
        print(command)
        os.system(command)
    paths = [tss_around_path,cleavage_around_path,donor_around_path,accept_around_path]
    for path in paths:
        read_path = "{}/{}_{}.bed".format(args.saved_root,bed_name,path)
        write_path = "{}/result_{}.bed".format(args.saved_root,path)
        write_bed(simply_coord(read_bed(read_path)),write_path)

    bed_id = set(list(read_bed(args.bed_path)['id']))
    data = {}
    with open(args.peptide_path) as fp:
        fasta_sequences = SeqIO.parse(fp, 'fasta')
        for fasta in fasta_sequences:
            if fasta.id in bed_id:
                data[fasta.id]=str(fasta.seq)

    with open(args.saved_root+'/peptide.fasta',"w") as fp:
        for id_,seq in data.items():
            fp.write(">"+str(id_)+"\n")
            fp.write(seq+"\n")

    for path in paths:
        read_path = "{}/result_{}.bed".format(args.saved_root,path)
        write_path = "{}/result_{}.fasta".format(args.saved_root,path)
        getfasta(args.genome_path,read_path,write_path)
        
    read_path = "{}/{}.bed".format(args.saved_root,bed_name)
    premRNA_write_path = "{}/{}.fasta".format(args.saved_root,bed_name)
    cDNA_write_path = "{}/{}_cDNA.fasta".format(args.saved_root,bed_name)
    getfasta(args.genome_path,read_path,premRNA_write_path)
    getfasta(args.genome_path,read_path,cDNA_write_path,use_split=True)