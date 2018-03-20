
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