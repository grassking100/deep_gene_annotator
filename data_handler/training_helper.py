from . import codes2vec
from . import numpy
from . import fasta2seqs
from . import deepdish
from . import random
from . import math
from . import DNA_SeqException
#read and return sequnece's one-hot-encoding vector and Exon-Intron data
def seqs2dnn_data(seqs,discard_dirty_sequence):
    code_dim=4
    vectors=[]
    valid_seqs_indice=[]
    index=0
    for seq in seqs:
        try:
            
            vec=codes2vec(seq)  
            vectors.append(numpy.array(vec).reshape(len(seq),code_dim))
            valid_seqs_indice.append(index)
            index+=1
        except DNA_SeqException as e:
            if not discard_dirty_sequence:
                raise e
    return (valid_seqs_indice,vectors)
#read and return sequnece's one-hot-encoding vector and annotation data
def seq_ann_alignment(fasta_path,annotation_path,discard_dirty_sequence):
    (names,seqs)=fasta2seqs(fasta_path)
    ann_types=['utr_5','utr_3','intron','cds','intergenic_region']
    #read annotation file
    ann_seqs=deepdish.io.load(annotation_path)
    anns=[]
    (valid_seqs_indice,seq_vecs)=seqs2dnn_data(seqs,discard_dirty_sequence)
    #for every name find corresponding sequnece and annotation 
    #and convert sequnece to one-hot-encoding vector
    for index in valid_seqs_indice:
        name=names[index]
        ann_seq=ann_seqs[str(name)]
        ann=[]
        for ann_type in ann_types:
            ann.append(ann_seq[ann_type])
        #append corresponding annotation to array
        anns.append(numpy.transpose(ann))
    return(seq_vecs,anns)

#get the index of cross validation data indice and testing data index
def data_index_splitter(data_number,fraction_of_traning_validation,number_of_cross_validation,shuffle=True):
    data_index=list(range( data_number))
    if shuffle:
        random.shuffle(data_index)
    traning_validation_number=(int)(data_number*fraction_of_traning_validation)
    train_validation_index=[data_index[i] for i in range(traning_validation_number)]
    testing_index=[data_index[i] for i in range(traning_validation_number,data_number)]
    cross_validation_index=[[] for i in range(number_of_cross_validation)]
    for i in range(traning_validation_number):
        cross_validation_index[i%number_of_cross_validation].append(train_validation_index[i])
    return(cross_validation_index,testing_index)
#check if sequnece is single exon by checking it is capital or not
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