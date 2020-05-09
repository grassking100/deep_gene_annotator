import os,sys
import math
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_fasta,write_fasta


def read_codon_table(path):
    df = pd.read_csv(path,comment='#',sep=' ',header=None)
    table = {}
    for row in df.values:
        index = 0
        while index < len(row):
            table[row[index]] = row[index+1]
            if row[index+3]=='i':
                index += 7
            else:
                index += 8
    return table


def get_orfs(cDNA):
    orfs = []
    for orf_index in range(3):
        orf = []
        length = math.floor((len(cDNA)-orf_index)/3)
        for index in range(0,length):
            codon = cDNA[orf_index+index*3:orf_index+index*3+3]
            orf.append(codon)
        orfs.append(orf)
    return orfs


def _get_longest_complete_orf(orf,start_codons,stop_codons):
    longest_orf_length = 0
    orf_indice = None
    start_codon_index = None
    for index,codon in enumerate(orf):
        if start_codon_index is None:
            if codon in start_codons:
                start_codon_index = index
        else:
            if codon in stop_codons:
                length = index - start_codon_index + 1
                if longest_orf_length < length:
                    longest_orf_length = length
                    orf_indice = (start_codon_index,index)
                start_codon_index = None
    if orf_indice is None:
        return None
    else:
        return orf_indice,orf[orf_indice[0]:orf_indice[1]+1]

    
def get_longest_complete_orf(cDNA,start_codons,stop_codons):
    longest_complete_orf_length = 0
    longest_complete_orf = None
    orf_indice = None
    for index,orf in enumerate(get_orfs(cDNA)):
        data = _get_longest_complete_orf(orf,start_codons,stop_codons)
        if data is not None:
            length = len(data[1])
            if longest_complete_orf_length < length:
                orf_indice,longest_complete_orf = data
                orf_indice = [orf_indice[0]*3+index,orf_indice[1]*3+index]
                longest_complete_orf_length = length
    if orf_indice is None:
        return None
    else:
        return orf_indice,longest_complete_orf


def get_start_stop_codon(table):
    start_codons = ['ATG']
    stop_codons = []
    for codon,aa in table.items():
        if aa == '*':
            stop_codons.append(codon)
    return start_codons,stop_codons


def translate(cDNA_fasta,codon_table_path=None):
    codon_table_path = codon_table_path or os.path.dirname(__file__) +'/../database/transl_table_1.txt'
    table = read_codon_table(codon_table_path)
    start_codons,stop_codons = get_start_stop_codon(table)
    orfs = {}
    peps = {}
    orf_indice_dict = {}
    for id_, cDNA in cDNA_fasta.items():
        data = get_longest_complete_orf(cDNA,start_codons,stop_codons)
        if data is not None:
            orf_indice,longest_complete_orf  = data
            orf_indice_dict[id_] = orf_indice
            orfs[id_] = ''.join(longest_complete_orf)
            aa_list = [table[codon] for codon in longest_complete_orf]
            peps[id_] = ''.join(aa_list)
    return orf_indice_dict,orfs,peps

def main(cDNA_fasta_path,orf_path=None,pep_path=None,orf_indice_path=None,codon_table_path=None):
    cDNA_fasta = read_fasta(cDNA_fasta_path)
    orf_indice_dict,orfs,peps = translate(cDNA_fasta,codon_table_path)
    if orf_indice_path is not None:            
        write_json(orf_indice_dict,orf_indice_path)
    if orf_path is not None:
        write_fasta(orfs,orf_path)
    if pep_path is not None:            
        write_fasta(peps,pep_path)
        

if __name__ == '__main__':
    parser = ArgumentParser(description='Get the longest complete ORF and its peptide')
    parser.add_argument("-i", "--cDNA_fasta_path", help="The cDNA fasta format", required=True)
    parser.add_argument("-o","--orf_path")
    parser.add_argument("-p","--pep_path")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
