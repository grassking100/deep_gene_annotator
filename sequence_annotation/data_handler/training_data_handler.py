"""This submodule provides API to handle data for training"""
import tensorflow as tf
import numpy
import deepdish
import random
from . import fasta2seqs
from . import seqs2dnn_data
def removed_terminal_tensors(true, pred, number_of_class, value_to_ignore):
    """Remove specific terminal singal"""
    reshape_pred = tf.reshape(pred, [-1])
    reshape_true = tf.reshape(true, [-1])
    index = tf.where(tf.not_equal(reshape_true, [value_to_ignore]))
    clean_pred = tf.reshape(tf.gather(reshape_pred, index), [-1, number_of_class])
    clean_true = tf.reshape(tf.gather(reshape_true, index), [-1, number_of_class])
    return (clean_true, clean_pred)


class SeqAnnAlignment():
    """Make data in fasta file can align with answer file"""
    def __init__(self, annotation_types):
        self.__names = []
        self.__seqs = []
        self.__anns = []
        self.__seqs_vecs = []
        self.__anns_count = {}
        self.__ANN_TYPES = annotation_types
        for ann_type in self.__ANN_TYPES:
            self.__anns_count[ann_type] = 0
    def add_file_to_parse(self, fasta_path, annotation_path, discard_dirty_sequence):
        """read and align sequnece's one-hot-encoding vector and annotation data"""
        (names, seqs) = fasta2seqs(fasta_path)
        self.__names += names
        self.__seqs += seqs
        #read annotation file
        ann_seqs = deepdish.io.load(annotation_path)
        (valid_seqs_indice, seqs_vecs) = seqs2dnn_data(seqs, discard_dirty_sequence)
        self.__seqs_vecs += seqs_vecs
        #for every name find corresponding sequnece and annotation
        #and convert sequnece to one-hot-encoding vector
        for index in valid_seqs_indice:
            name = names[index]
            ann_seq = ann_seqs[str(name)].tolist()
            ann = []
            for ann_type in self.ANN_TYPES:
                temp = ann_seq[ann_type]
                self.__anns_count[ann_type] += numpy.sum(temp)
                ann.append(temp)
            #append corresponding annotation to array
            self.__anns.append(numpy.transpose(ann))
    @property
    def ANN_TYPES(self):
        """Get annotation type"""
        return self.__ANN_TYPES
    @ANN_TYPES.setter
    def ANN_TYPES(self, ANN_TYPES):
        """Set annotation type"""
        self.__ANN_TYPES = ANN_TYPES
    @property
    def names(self):
        """Get names"""
        return self.__names
    @property
    def seqs_vecs(self):
        """Get seqeunces in vector format"""
        return self.__seqs_vecs
    @property
    def seqs_annotations(self):
        """Get seqeunces's annotation in vector format"""
        return self.__anns
    @property
    def seqs_annotations_count(self):
        """Get annotation count"""
        return self.__anns_count
def handle_alignment_files(fasta_files_path, answer_file_path,ANNOTATION_TYPES):
    """Make data in many fasta files can align with answer file"""
    alignment = SeqAnnAlignment(ANNOTATION_TYPES)
    for file in fasta_files_path:
        alignment.add_file_to_parse(file, answer_file_path, True)
    return (alignment.seqs_vecs, alignment.seqs_annotations, alignment.seqs_annotations_count)

def data_index_splitter(data_number, fraction_of_traning_validation,
                        number_of_cross_validation, shuffle=True):
    """get the index of cross validation data indice and testing data index"""
    data_index = list(range(data_number))
    if shuffle:
        random.shuffle(data_index)
    traning_validation_number = (int)(data_number*fraction_of_traning_validation)
    train_validation_index = [data_index[i] for i in range(traning_validation_number)]
    testing_index = [data_index[i] for i in range(traning_validation_number, data_number)]
    cross_validation_index = [[] for i in range(number_of_cross_validation)]
    for i in range(traning_validation_number):
        cross_validation_index[i%number_of_cross_validation].append(train_validation_index[i])
    return(cross_validation_index, testing_index)

def is_single_exon(seq):
    """check if sequnece is single exon by checking if it is capital or not"""
    for character in seq:
        if not character.isupper():
            return False
    return True

def seqs_index_selector(seqs, min_length, max_length, exclude_single_exon):
    """
        select sequnece index which length is between the specific
        range and choose to exclude single exon or not
    """
    sub_index = []
    lengths = [len(s) for s in seqs]
    if max_length == -1:
        max_length = max(lengths)
    for i in range(len(lengths)):
        if lengths[i] >= min_length and lengths[i] <= max_length:
            sub_index.append(i)
    target_index = []
    if exclude_single_exon:
        for i in sub_index:
            if not is_single_exon(seqs[i]):
                target_index.append(i)
    else:
        target_index = sub_index
    return target_index
