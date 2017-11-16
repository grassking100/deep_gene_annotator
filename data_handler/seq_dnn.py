from . import codes2vec
from . import numpy
from . import fasta2arr
from . import deepdish
#read and return sequnece's one-hot-encoding vector and Exon-Intron data
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

#read and return sequnece's one-hot-encoding vector and annotation data
def seq_ann_alignment(fasta_path,annotation_path,safe):
    (names,seqs)=fasta2arr(fasta_path)
    ann_types=['utr_5','utr_3','intron','cds','intergenic_region']
    #read annotation file
    ann_seqs=deepdish.io.load(annotation_path)
    anns=[]
    seq_vecs=[]
    #for every name find corresponding sequnece and annotation 
    #and convert sequnece to one-hot-encoding vector
    for name,seq in zip(names,seqs):
        seq_vecs.append(codes2vec(seq,safe))
        ann_seq=ann_seqs[str(name)]
        ann=[]
        for ann_type in ann_types:
            ann.append(ann_seq[ann_type])
        #append corresponding annotation to array
        anns.append(numpy.transpose(ann))
    return(seq_vecs,anns)