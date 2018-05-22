import random
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
