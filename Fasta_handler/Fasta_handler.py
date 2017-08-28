from . import SeqIO
from . import numpy
from . import codes2vec
#read fasta file and return array of sequnece and name
#if number is negative,then all the sequneces will be read
#otherwirse read part of sequneces,the number indicate how many to read
def fastas2arr(file_name,number=-1):
    fasta_sequences = SeqIO.parse(open(file_name),'fasta')
    names=[]
    seqs=[]
    counter=0
    for fasta in fasta_sequences:
        if (number<=0)|(counter<number): 
            name,seq=fasta.id,(str)(fasta.seq)
            names.append(name)
            seqs.append(seq)
            counter+=1
        else:
            break
    return(names,seqs)
#read fasta file
#and return the data format which tensorflow can input
def fastas2dnn_data(file_name,number=-1,safe=False):
    (names,seqs)=fastas2arr(file_name,number)
    return seqs2dnn_data(seqs,safe)
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
#check if sequnece is single exon
def is_single_exon(seq):
    for s in seq:
        if not s.isupper():
            return False
    return True
#select sequnece index which length is between the specific range and choose to exclude single exon or not
def seqs_index_selector(seqs,min_length,max_length,exclude_single_exon):
    sub_index=[]
    length=[len(s) for s in seqs]
    if max_length==-1:
          max_length=max(length)
    for i in range(len(length)):
        if length[i]>=min_length and length[i]<=max_length:
            sub_index.append(i)
    target_index=[]
    if exclude_single_exon:
        for i in sub_index:
            if not is_single_exon(seqs[i]):
                target_index.append(i)
    else:
        target_index=sub_index
    return target_index