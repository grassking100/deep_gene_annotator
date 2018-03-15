"""
    read fasta file and return array of sequnece and name
    if number is negative, then all the sequneces will be read
    otherwirse read part of sequneces, the number indicate how many to read
"""
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet.IUPAC import IUPACAmbiguousDNA
from .sequence_handler import seqs2dnn_data

def fasta2seqs(file_name, number=-1):
    """Change sequences in fasta file into dictionary of seqeucnces """
    fasta_sequences = SeqIO.parse(open(file_name), 'fasta')
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

def fasta2dnn_data(file_name, number=-1, safe=False):
    """read fasta file and return the data format which tensorflow can input"""
    seqs = fasta2seqs(file_name, number).values()
    return seqs2dnn_data(seqs, safe)

class FastaExtractor:
    """Read data in fasta file and stored in list format"""
    def __init__(self, fasta_file):
        self.__records = []
        data = fasta2seqs(fasta_file)
        for name, seq in data.items():
            temp = SeqRecord(Seq(seq, IUPACAmbiguousDNA), id=name)
            self.__records.append(temp)
    def get_record(self, indice):
        """Get record by index"""
        return [self.__records[i] for i in indice]
    def save_as_fasta(self, indice, file_path):
        """Save list format data into fasta file"""
        records = self.get_record(indice)
        with open(file_path, "w") as output_handle:
            for record in records:
                SeqIO.write(record, output_handle, "fasta")
