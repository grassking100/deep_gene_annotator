from . import codes2vec
from . import numpy
#read sequnece
#and return the data format which tensorflow can input
def seqs2dnn_data(seqs,safe=False):
    code_dim=4
    xs=[]
    ys=[]
    for seq in seqs:
        vec=codes2vec(seq,safe)
        if vec is not None:
            x =numpy.array(vec).reshape(len(seq),code_dim)
            y =[(float)(s.isupper()) for s in seq]
            y=numpy.array(y).reshape(len(y),1)
            xs.append(x)
            ys.append(y)
    return(xs, ys)
