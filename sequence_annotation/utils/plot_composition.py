import os, sys
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_bed, read_fasta
from sequence_annotation.utils.seq_converter import SeqConverter,DNA_CODES

def fasta_composition(fasta_data,converter=None):
    if converter is None:
        converter = SeqConverter()
    sum_ = None
    for seq in fasta_data.values():
        vecs = np.array(converter.seq2vecs(seq))
        if sum_ is None:
            sum_ = vecs
        else:
            sum_ += vecs
    return sum_/sum_.sum(1)[0]
            
def plot_composition(composition,output_path,
                     title=None,x_shift=None,xlabel=None):
    title = title or ''
    xlabel = xlabel or "From 5' to 3'"
    x_shift= x_shift or 0
    range_ = list(range(x_shift,x_shift+composition.shape[0]))
    fig, ax = plt.subplots()
    for seq,label in zip(composition.T,list(DNA_CODES)):
        ax.plot(range_,seq,label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ratio")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.legend()
    fig.savefig(output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",help='Fasta path',required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("--x_shift",type=int)
    parser.add_argument("--xlabel")
    parser.add_argument("--title")

    args = parser.parse_args()
    
    fasta = read_fasta(args.input_path)
    composition = fasta_composition(fasta)
    plot_composition(composition,args.output_path,
                     title=args.title,x_shift=args.x_shift,
                     xlabel=args.xlabel)
    