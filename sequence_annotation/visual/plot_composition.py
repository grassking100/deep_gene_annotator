import os, sys
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import AutoMinorLocator
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import read_fasta
from sequence_annotation.utils.seq_converter import SeqConverter, DNA_CODES


def get_alphabet_color_map():
    css4_color_names = sorted(mcolors.CSS4_COLORS.keys())
    basic_color_map = {'A': 'g', 'T': 'r', 'C': 'b', 'G': 'y'}
    color_map = {}
    for index in range(97, 123):
        char = chr(index).upper()
        if char in basic_color_map.keys():
            color = basic_color_map[char]
        else:
            color = css4_color_names[index]
        color_map[char] = color
    return color_map


def fasta_composition(converter, fasta_data, truncated=None):
    truncated = truncated or 0
    sum_ = None
    for seq in fasta_data.values():
        if truncated != 0:
            seq = seq[truncated:-truncated]
        vecs = np.array(converter.seq2vecs(seq))
        if sum_ is None:
            sum_ = vecs
        else:
            sum_ += vecs
    return sum_ / sum_.sum(1)[0]


def plot_composition(composition, output_path, title=None, 
                     xlabel=None, shift=None, truncated=None,
                     codes=None):

    truncated = truncated or 0
    codes = codes or DNA_CODES
    title = title or ''
    xlabel = xlabel or "From 5' to 3'"
    shift = shift or 0
    range_ = list(range(shift, shift + composition.shape[0]))
    alphabet_color_map = get_alphabet_color_map()
    fig, ax = plt.subplots()
    for seq, label in zip(composition.T, list(codes)):
        if sum(seq) > 0:
            ax.plot(range_, seq, label=label, color=alphabet_color_map[label])
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Ratio", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right",fontsize=16)
    fig.savefig(output_path, bbox_inches = 'tight',pad_inches = 0)


def main(input_path, output_path, truncated=None, **kwargs):
    fasta = read_fasta(input_path)
    codes = set()
    for seq in fasta.values():
        codes = codes.union(set(seq))
    codes = sorted(list(codes))
    converter = SeqConverter(codes=codes)
    composition = fasta_composition(converter, fasta, truncated=truncated)
    plot_composition(composition, output_path, codes=codes, **kwargs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", help='Fasta path', required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-s", "--shift", type=int)
    parser.add_argument("-t", "--truncated", type=int, default=0)
    parser.add_argument("-x", "--xlabel")
    parser.add_argument("-n", "--title")

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
