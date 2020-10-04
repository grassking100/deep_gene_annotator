import pytz
import datetime
import re
import sys
import os
import errno
import math
import json
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')


class CONSTANT_LIST(list):
    def __init__(self, list_):
        super().__init__(list_)

    def __delitem__(self, index):
        raise Exception("{} cannot delete element".format(
            self.__class__.__name__))

    def insert(self, index, value):
        raise Exception("{} cannot insert element".format(
            self.__class__.__name__))

    def __setitem__(self, index, value):
        raise Exception("{} cannot set element".format(
            self.__class__.__name__))


class CONSTANT_DICT(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        raise Exception("{} cannot set element".format(
            self.__class__.__name__))

    def __delitem__(self, key):
        raise Exception("{} cannot delete element".format(
            self.__class__.__name__))


def logical_not(lhs, rhs):
    return np.logical_and(lhs, np.logical_not(rhs))


def create_folder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as erro:
        if erro.errno != errno.EEXIST:
            raise


def copy_path(root, path):
    target_path = os.path.join(root, path.split('/')[-1])
    if not os.path.exists(target_path):
        command = 'cp -t {} {}'.format(root, path)
        os.system(command)


def get_protected_attrs_names(object_):
    class_name = object_.__class__.__name__
    attrs = [
        attr for attr in dir(object_)
        if attr.startswith('_') and not attr.endswith('__')
        and not attr.startswith('_' + class_name + '__')
    ]
    return attrs


def split(ids, ratios):
    if round(sum(ratios)) != 1:
        raise Exception("Ratio sum should be one")
    lb = ub = 0
    ids_list = []
    id_len = len(ids)
    sum_ = 0
    for ratio in ratios:
        ub += ratio
        item = ids[math.ceil(lb * id_len):math.ceil(ub * id_len)]
        sum_ += len(item)
        ids_list.append(item)
        lb = ub
    if sum_ != id_len:
        raise Exception("Id number is not consist with origin count")
    return ids_list


def get_subdict(ids, data):
    return dict(zip(ids, [data[id_] for id_ in ids]))


def print_progress(info):
    print(info, end='\r')
    sys.stdout.write('\033[K')


def write_json(json_, path, mode=None):
    mode = mode or 'w'
    with open(path, mode) as fp:
        json.dump(json_, fp, indent=4, sort_keys=True)


def read_json(path, mode=None):
    mode = mode or 'r'
    with open(path, mode) as fp:
        return json.load(fp)


def replace_utc_to_local(timestamp, timezone_=None):
    timezone_ = timezone_ or pytz.timezone('ROC')
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=pytz.timezone('UTC'))
    return timestamp.astimezone(timezone_)


def get_time(timezone_=None):
    timezone_ = timezone_ or pytz.timezone('ROC')
    time_data = datetime.datetime.now(timezone_)
    return time_data


def to_time_str(time_data):
    time_str = time_data.strftime("%Y-%m-%d %H:%M:%S.%f %z")
    return time_str


def from_time_str(time_str):
    time_data = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f %z")
    return time_data


def get_time_str(timezone_=None):
    time_str = to_time_str(get_time(timezone_))
    return time_str


def get_file_name(path, with_postfix=False):
    name = path.split('/')[-1]
    if not with_postfix:
        name = '.'.join(name.split('.')[:-1])
    return name


def batch_join(project_root, folder_names, path):
    paths = {}
    for folder_name in folder_names:
        paths[folder_name] = os.path.join(project_root, folder_name, path)
    return paths


def find_substr(regex, string, shift_value=None):
    """Find indice of matched text in string

    Parameters:
    ----------
    regex : str
        Regular expression
    string : str
        String to be searched
    shift_value : int (default: 0)
        Shift the index

    Returns:
    ----------
    list (int)
        List of indice
    """
    shift_value = shift_value or 0
    iter_ = re.finditer(regex, string)
    indice = [m.start() + shift_value for m in iter_]
    return indice

def get_subdict(dict_,ids):
    intersected_ids = set(ids).intersection(list(dict_.keys()))
    subdict = {}
    for id_ in intersected_ids:
        subdict[id_] = dict_[id_]
    return subdict
