"""
    read fasta file and return array of sequnece and name
    if number is negative, then all the sequneces will be read
    otherwirse read part of sequneces, the number indicate how many to read
"""
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet.IUPAC import IUPACAmbiguousDNA
import numpy as np
from . import SeqConverter
from . import SeqException
class FastaConverter():
    def __init__(self,seq_converter=None):
        self._seq_converter = seq_converter or SeqConverter()
    def to_seq_dict(self,fasta_path,number=-1):
        """Change sequences in fasta file into dictionary of seqeucnces """
        fasta_sequences = SeqIO.parse(open(fasta_path), 'fasta')
        data = {}
        counter = 0
        for fasta in fasta_sequences:
            if (number <= 0) or (counter < number):
                name, seq = fasta.id, (str)(fasta.seq)
                data[name]=seq
                counter += 1
            else:
                break
        return data
    def to_vec_dict(self,seq_dict, discard_invalid_seq, number=-1):
        """convert dictionary of seqeucnces to dictionary of one-hot encoding vector"""
        data = {}
        for name,seq in seq_dict.items():
            try:
                data[name] = self._seq_converter.seq2vecs(seq)
            except SeqException as exp:
                if not discard_invalid_seq:
                       raise exp
        return data
