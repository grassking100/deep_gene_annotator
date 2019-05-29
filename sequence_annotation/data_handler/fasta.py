"""
    Tools to hadnler with fatsa file
"""
from Bio import SeqIO
from pathlib import Path
import subprocess

def read_fasta(paths):
    """Read fasta file and return dictionary of sequneces"""
    if not isinstance(paths,list):
        paths = [paths]
    data = {}
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        with open(path) as file:
            fasta_sequences = SeqIO.parse(file, 'fasta')
            for fasta in fasta_sequences:
                name, seq = fasta.id, (str)(fasta.seq)
                data[name]=seq
    return data

def write_fasta(path,seqs):
    """Read dictionary of sequneces into fasta file"""
    with open(path,"w") as file:
        for id_,seq in seqs.items():
            file.write(">" + id_ + "\n")
            file.write(seq + "\n")

def get_fasta(gtf,fasta,saved_name):
    command = 'bedtools getfasta -s -name -bed '+gtf+' -fi '+fasta+' -fo '+saved_name
    return subprocess.call(command, shell=True)