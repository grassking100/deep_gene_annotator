from ..utils.utils import get_file_name


def get_dl_folder_names(split_table):
    dl_folder_names = []
    for item in split_table.to_dict('record'):
        folder_name = "{}_{}".format(get_file_name(item['training_path']),
                                     get_file_name(item['validation_path']))
        dl_folder_names.append(folder_name)
    return dl_folder_names


def get_augustus_folder_names(aug_folder_prefix, num):
    aug_folder_names = []
    for index in range(num):
        folder_name = "{}_{}".format(aug_folder_prefix, index + 1)
        aug_folder_names.append(folder_name)
    return aug_folder_names


def _get_min_threshold(params, alpha=None):
    threshold = None
    alpha = alpha or 4
    weights = params['weights']
    means = params['means']
    stds = params['stds']
    for index in range(len(weights)):
        mean = means[index]
        std = stds[index]
        new_threshold = mean - std * alpha
        if threshold is None:
            threshold = new_threshold
        else:
            threshold = min(new_threshold, threshold)
    return threshold


def get_min_threshold(params, alpha=None):
    threshold = pow(10, _get_min_threshold(params, alpha))
    return threshold
