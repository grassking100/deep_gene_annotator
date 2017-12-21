from . import codes2vec
from . import numpy
from . import fasta2seqs
from . import deepdish
from . import random
from . import math
from . import DNA_SeqException
def seqs2dnn_data(seqs,discard_dirty_sequence):
    #read and return sequnece's one-hot-encoding vector and Exon-Intron data
    code_dim=4
    vectors=[]
    valid_seqs_indice=[]
    for seq,index in zip(seqs,range(len(seqs))):
        try:
            vec=codes2vec(seq)  
            vectors.append(numpy.array(vec).reshape(len(seq),code_dim))
            valid_seqs_indice.append(index)
        except DNA_SeqException as e:
            if not discard_dirty_sequence:
                raise e
    return (valid_seqs_indice,vectors)

class SeqAnnAlignment():
    def __init__(self,annotation_types):
        self.__names=[]
        self.__seqs=[]
        self.__anns=[]
        self.__seqs_vecs=[]
        self.__anns_count={}
        self.ANN_TYPES=annotation_types
        for ann_type in self.ANN_TYPES:
                self.__anns_count[ann_type]=0
    def add_file_to_parse(self,fasta_path,annotation_path,discard_dirty_sequence):
        #read and return sequnece's one-hot-encoding vector and annotation data
        (names,seqs)=fasta2seqs(fasta_path)
        self.__names+=names
        self.__seqs+=seqs
        #read annotation file
        ann_seqs=deepdish.io.load(annotation_path)
        (valid_seqs_indice,seqs_vecs)=seqs2dnn_data(seqs,discard_dirty_sequence)
        self.__seqs_vecs+=seqs_vecs
        #for every name find corresponding sequnece and annotation 
        #and convert sequnece to one-hot-encoding vector
        for index in valid_seqs_indice:
            name=names[index]
            ann_seq=ann_seqs[str(name)].tolist()
            ann=[]
            for ann_type in self.ANN_TYPES:
                temp=ann_seq[ann_type]
                self.__anns_count[ann_type]+=numpy.sum(temp)
                ann.append(temp)
            #append corresponding annotation to array
            self.__anns.append(numpy.transpose(ann))
    @property
    def ANN_TYPES(self):
        return self.__ANN_TYPES         
    @ANN_TYPES.setter
    def ANN_TYPES(self,v):
        self.__ANN_TYPES=v
    @property
    def names(self):
        return self.__names
    @property
    def seqs_vecs(self):
        return self.__seqs_vecs
    @property
    def seqs_annotations(self):
        return self.__anns
    @property
    def seqs_annotations_count(self):
        return self.__anns_count


def data_index_splitter(data_number,fraction_of_traning_validation,number_of_cross_validation,shuffle=True):
    #get the index of cross validation data indice and testing data index
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

def is_single_exon(seq):
    #check if sequnece is single exon by checking it is capital or not
    for s in seq:
        if not s.isupper():
            return False
    return True

def seqs_index_selector(seqs,min_length,max_length,exclude_single_exon):
    #select sequnece index which length is between the specific range and choose to exclude single exon or not
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