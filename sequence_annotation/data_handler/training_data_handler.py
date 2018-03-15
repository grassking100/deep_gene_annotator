"""This submodule provides API to handle data for training"""
import tensorflow as tf
import numpy as np
import random
from . import fasta2seqs
from . import seqs2dnn_data
from . import LengthNotEqualException
def process_tensor(true, pred,values_to_ignore=None):
    """Remove specific terminal singal and concatenate them into a two dimension tensor"""
    reshape_pred = tf.reshape(pred, [-1])
    reshape_true = tf.reshape(true, [-1])
    if values_to_ignore is not None:
        if not isinstance(values_to_ignore,list):
            values_to_ignore=[values_to_ignore]
        index = tf.where(tf.not_equal(reshape_true, values_to_ignore))
        clean_pred = tf.reshape(tf.gather(reshape_pred, index), [-1, tf.shape(pred)[2]])
        clean_true = tf.reshape(tf.gather(reshape_true, index), [-1, tf.shape(true)[2]])
    else:
        clean_pred = tf.reshape(pred, [-1, tf.shape(pred)[2]])
        clean_true = tf.reshape(true, [-1, tf.shape(true)[2]])
    return (clean_true, clean_pred)

class SeqAnnAlignment():
    """Make data in fasta file can align with answer file"""
    def __init__(self):
        self._names = []
        self._seqs = []
        self._ann_vecs = []
        self._seqs_vecs = []
        self._ann_count = {}
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
        seq_dict = fasta2seqs(fasta_path)    
        seq_vec_dict = seqs2dnn_data(seq_dict,discard_dirty_sequence)
        #read annotation file
        ann_seqs = self._load_ann_data(annotation_path)
        ann_count = {}
        ann_vecs = []
        seqs_vecs = []
        names = []
        seqs = []
        #for every name find corresponding sequnece and annotation
        #and convert sequnece to one-hot-encoding vector
        for name,seq_vec in seq_vec_dict.items():
            ann_seq = ann_seqs[name]
            seq = seq_dict[name]
            seqs_vecs.append(seq_vec_dict[name])
            names.append(name)
            seqs.append(seq)
            ann = []
            for ann_type,value in ann_seq.items():
                if ann_type not in ann_count.keys():
                    ann_count[ann_type] = 0
                ann_count[ann_type] += np.sum(value)
                ann.append(value)
            #append corresponding annotation to array
            transposed_ann = np.transpose(ann)
            ann_length = np.shape(transposed_ann)[0]
            seq_length = np.shape(seq_vec)[0]
            if ann_length != seq_length:
                print(index)
                raise LengthNotEqualException(ann_length, seq_length)
            ann_vecs.append(transposed_ann)
        for ann_type, count in ann_count.items():
            if ann_type not in self._ann_count.keys():
                self._ann_count[ann_type] = 0
            self._ann_count[ann_type] += ann_count[ann_type]
        self._ann_vecs += ann_vecs
        self._names += names
        self._seqs += seqs
        self._seqs_vecs += seqs_vecs
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
    def seqs_ann_vecs(self):
        """Get seqeunces's annotation in vector format"""
        return self._ann_vecs
    @property
    def seqs_ann_count(self):
        """Get annotation count"""
        return self._ann_count
def handle_alignment_files(fasta_file_patha, answer_file_path):
    """Make data in many fasta files can align with answer file"""
    alignment = SeqAnnAlignment()
    for file in fasta_file_patha:
        alignment.parse_file(file, answer_file_path, True)
    return (alignment.seqs_vecs, alignment.seqs_ann_vecs, alignment.seqs_ann_count)

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
