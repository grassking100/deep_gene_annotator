"""
    read fasta file and return array of sequnece and name
    if number is negative, then all the sequneces will be read
    otherwirse read part of sequneces, the number indicate how many to read
"""
from Bio import SeqIO
from . import SeqConverter
from . import SeqException
class FastaConverter():
    def __init__(self,seq_converter=None):
        self._seq_converter = seq_converter or SeqConverter()
    def to_seq_dict(self,fasta_paths,number=-1):
        """Change sequences in fasta file into dictionary of seqeucnces """
        if not isinstance(fasta_paths,list):
            fasta_paths = [fasta_paths]
        data = {}
        counter = 0
        for fasta_path in fasta_paths:
            with open(fasta_path) as file:
                fasta_sequences = SeqIO.parse(file, 'fasta')
                for fasta in fasta_sequences:
                    if (number <= 0) or (counter < number):
                        name, seq = fasta.id, (str)(fasta.seq)
                        data[name]=seq
                        counter += 1
                    else:
                        break
        return data
    def to_vec_dict(self,seq_dict, discard_invalid_seq):
        """convert dictionary of seqeucnces to dictionary of one-hot encoding vector"""
        data = {}
        for name,seq in seq_dict.items():
            try:
                data[name] = self._seq_converter.seq2vecs(seq)
            except SeqException as exp:
                if not discard_invalid_seq:
                    raise exp
        return data
