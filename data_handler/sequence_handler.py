"""This submodule will handler DNA sequence and one-hot encoded seqeunce convert"""
import numpy
class DNACodeException(Exception):
    """Raise when input character is not in in defined space"""
    pass
class DNASeqException(Exception):
    """Raise wehn input sequences has at least a chacarter is not in in defined space"""
    pass
def code2vec(code):
    """convert DNA code to one hot encoding"""
    target = ['A', 'T', 'C', 'G']
    nucleotide_a = [1, 0, 0, 0]
    nucleotide_t = [0, 1, 0, 0]
    nucleotide_c = [0, 0, 1, 0]
    nucleotide_g = [0, 0, 0, 1]
    vec = [nucleotide_a, nucleotide_t, nucleotide_c, nucleotide_g]
    length_of_nucleotide_type = len(target)
    for i in range(length_of_nucleotide_type):
        if code.upper() == target[i]:
            return vec[i]
    raise DNACodeException(str(code)+' is not in space')
def vec2code(vector):
    """convert one hot encoding to DNA code"""
    nucleotide_a = [1, 0, 0, 0]
    nucleotide_t = [0, 1, 0, 0]
    nucleotide_c = [0, 0, 1, 0]
    nucleotide_g = [0, 0, 0, 1]
    target = [nucleotide_a, nucleotide_t, nucleotide_c, nucleotide_g]
    code = ['A', 'T', 'C', 'G']
    length_of_nucleotide_type = len(target)
    for i in range(length_of_nucleotide_type):
        if vector == target[i]:
            return code[i]
    raise DNACodeException(str(vector)+' is not in space')

def codes2vec(codes):
    """convert DNA sequence to one hot encoding sequence"""
    code_list = list(codes)
    arr = []
    for code in code_list:
        try:
            arr.append(code2vec(code))
        except DNACodeException:
            raise DNASeqException('Sequence has invalid code in it')
    return arr

def vec2codes(vector):
    """convert one hot encoding sequence to DNA sequence"""
    characters = list(vector)
    arr = []
    for character in characters:
        try:
            arr.append(vec2code(character))
        except DNACodeException:
            raise DNASeqException('Sequence vector has invalid vector in it')
    return arr
def seqs2dnn_data(seqs, discard_dirty_sequence):
    """read and return valid indice and sequnece's one-hot-encoding vector"""
    code_dim = 4
    vectors = []
    valid_seqs_indice = []
    for seq, index in zip(seqs, range(len(seqs))):
        try:
            vec = codes2vec(seq)
            vectors.append(numpy.array(vec).reshape(len(seq), code_dim))
            valid_seqs_indice.append(index)
        except DNASeqException as exception:
            if not discard_dirty_sequence:
                raise exception
    return (valid_seqs_indice, vectors)
