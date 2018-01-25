"""This module includes librarys which can handle data"""
import random
import math
import numpy
import deepdish
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet.IUPAC import IUPACAmbiguousDNA
from .sequence_handler import codes2vec, seqs2dnn_data
from .fasta_handler import fasta2seqs
from .training_data_handler import seqs2dnn_data

#from .sequence_handler import codes2vec, DNASeqException, seqs2dnn_data

