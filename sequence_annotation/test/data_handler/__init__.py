import os
import sys
sys.path.append(os.path.abspath(__file__+"/../../../../"))
from sequence_annotation.data_handler.sequence_handler import code2vec,vec2code,codes2vec,vec2codes,seqs2dnn_data,DNASeqException