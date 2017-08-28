from . import random
from . import seqs2dnn_data
from . import numpy
#selection portion of index to be training data index and other become validation data index
def traning_validation_data_index_selector(seqs_index,fraction_of_traning,shuffle=True):
    copy_index=[i for i in seqs_index]
    if shuffle:
        random.shuffle(copy_index)
    half_number=(int)(len(copy_index)*fraction_of_traning)
    train_index=[i for i in copy_index[:half_number]]
    validation_index=[i for i in copy_index[half_number:]]
    return(train_index,validation_index)