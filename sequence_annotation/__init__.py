import sys
import os
sys.path.append(os.path.expanduser('~/../home/BatchRenormalization'))
sys.path.append(os.path.expanduser('~/../home/keras-minimal-rnn'))
sys.path.append(os.path.expanduser('~/../home/Keras-IndRNN'))
sys.path.append(os.path.expanduser('~/../home/MReluGRU'))
from minimal_rnn import MinimalRNN
from ind_rnn import IndRNN
from batch_renorm import BatchRenormalization
from recurrent_bot import MReluGRU