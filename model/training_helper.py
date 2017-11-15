from . import random
from . import seqs2dnn_data
from . import numpy
#selection portion of index to be training data index and other become validation data index
def traning_validation_data_index_selector(seqs_index,fraction_of_traning,shuffle=True):
    copy_index=[i for i in seqs_index]
    if shuffle:
        random.shuffle(copy_index)
    half_number=(int)(len(copy_index)*fraction_of_traning)
    train_index=[i for i in copy_index[:half_number]]
    validation_index=[i for i in copy_index[half_number:]]
    return(train_index,validation_index)
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