import sys
import os
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_fasta

def main(fasta_path,output_path):
    fasta = read_fasta(fasta_path)
    seqs = []
    for id_,seq in fasta.items():
        seqs.append({'id':id_,'motif':seq})
    seqs = pd.DataFrame.from_dict(seqs)
    counts = seqs['motif'].value_counts().to_frame()
    counts.index.name = 'motif'
    counts.columns = ['count']
    counts.to_csv(output_path,sep='\t',header=True)

if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--fasta_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    args = parser.parse_args()

    kwargs = vars(args)
    main(**kwargs)
