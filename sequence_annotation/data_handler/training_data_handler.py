"""This submodule provides API to handle data for training"""
import tensorflow as tf
import numpy as np
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
    def __init__(self):
        self._names = []
        self._seqs = []
        self._anns = []
        self._seqs_vecs = []
        self._anns_count = {}
        self._ann_data_path = []
        self._ann_data = None
    def _load_ann_data(self, ann_data_path):
        if ann_data_path is not self._ann_data_path:
            self._ann_data_path=ann_data_path
            self._ann_data = np.load(ann_data_path).item()
        return self._ann_data
    def parse_file(self, fasta_path, annotation_path, discard_dirty_sequence):
        """
            read and align sequnece's one-hot-encoding vector 
            and annotation data ,then it stores data
        """
        (names, seqs) = fasta2seqs(fasta_path)
        self._names += names
        self._seqs += seqs
        (valid_seqs_indice, seqs_vecs) = seqs2dnn_data(seqs, discard_dirty_sequence)
        self._seqs_vecs += seqs_vecs
        #read annotation file
        ann_seqs = self._load_ann_data(annotation_path)
        #for every name find corresponding sequnece and annotation
        #and convert sequnece to one-hot-encoding vector

        for index in valid_seqs_indice:
            name = names[index]
            ann_seq = ann_seqs[str(name)]
            ann = []
            for ann_type,value in ann_seq.items():
                if ann_type not in self._anns_count.keys():
                    self._anns_count[ann_type] = 0
                self._anns_count[ann_type] += np.sum(value)
                ann.append(value)
            #append corresponding annotation to array
            self._anns.append(np.transpose(ann))
    @property
    def ANN_TYPES(self):
        """Get annotation type"""
        return self._anns_count.keys()
    @property
    def names(self):
        """Get names"""
        return self._names
    @property
    def seqs_vecs(self):
        """Get seqeunces in vector format"""
        return self._seqs_vecs
    @property
    def seqs_annotations(self):
        """Get seqeunces's annotation in vector format"""
        return self._anns
    @property
    def seqs_ann_count(self):
        """Get annotation count"""
        return self._anns_count
def handle_alignment_files(fasta_file_patha, answer_file_path):
    """Make data in many fasta files can align with answer file"""
    alignment = SeqAnnAlignment()
    for file in fasta_file_patha:
        alignment.parse_file(file, answer_file_path, True)
    return (alignment.seqs_vecs, alignment.seqs_annotations, alignment.seqs_ann_count)

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
