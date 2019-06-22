from math import log
from matplotlib import pyplot as plt

def seq2logo(seq,symbols,expected=None):
    if expected is None:
        expected = []
        for symbol in symbols:
            expected[symbol] = 1/len(symbols)
    logo = []
    for layer in seq:
        nucleotide = []
        for val,bg in zip(layer,expected):
            nucleotide.append(log((val+1e-20)/bg,2)*val)
        nucleotide = np.array(nucleotide)
        nucleotide = layer * sum(nucleotide)
        logo.append([(cha,val) for val ,cha in zip(nucleotide,symbols)])
    return logo

def composition_count(seqs,L=100):
    chars = set()
    length = None
    for seq in seqs:
        if length is None:
            length = len(seq)
        else:
            if length != len(seq):
                raise Exception("Sequences length should be same")
        if len(seq) < L:
            raise Exception("Sequence length should not smaller than "+str(L))
        chars = chars.union(set(list(seq)))
    count = {}
    stats = {}
    for char in chars:
        count[char] = [0]*L
        stats[char] = [0]*L
    for seq in seqs:
        length = len(seq)
        for index,char in enumerate(list(seq)):
            count[char][int(index*L/length)] += 1
    sum_ = [0]*L    
    for index in range(L):
        for char in chars:
            sum_[index] += count[char][index]
    for index in range(L):
        for char in chars:
            if sum_[index]!=0:
                stats[char][index] = count[char][index]/sum_[index]
    return stats,chars

def  composition_plot(seqs,L):
    stats,chars = composition_count(seqs,L=L)
    for char in chars:
        plt.plot(stats[char],label=char)
    plt.legend()