from math import log
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