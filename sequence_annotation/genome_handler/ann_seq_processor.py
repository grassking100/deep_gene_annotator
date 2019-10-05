import warnings
import numpy as np
import re
from .exception import ProcessedStatusNotSatisfied,InvalidStrandType
from .sequence import AnnSequence

def is_dirty(seq, dirty_types):
    for dirty_type in dirty_types:
        if np.sum(seq.get_ann(dirty_type)) > 0:
            return True
    return False

def get_certain_status(seq, focus_types=None):
    if not seq.processed_status == 'normalized':
        raise ProcessedStatusNotSatisfied(seq.processed_status,'normalized')
    focus_types = focus_types or seq.ANN_TYPES
    ann = seq.to_dict(only_data=True)
    data = [ann[type_] for type_ in focus_types]
    certain_status = np.ceil(np.array(data)).sum(axis=0)==1
    return np.array(certain_status,dtype='bool')

def _get_unfocus_types(ann_types,focus_types):
    other_type = list(set(ann_types).difference(set(focus_types)))
    return other_type

def get_normalized(seq, focus_types=None):
    if seq.processed_status=='normalized':
        return seq
    else:
        focus_types = focus_types or seq.ANN_TYPES
        if not is_full_annotated(seq,focus_types):
            raise Exception("Sequence is not fully annotated")
        focus_types = focus_types or seq.ANN_TYPES
        other_types = _get_unfocus_types(seq.ANN_TYPES,focus_types)
        norm_seq = AnnSequence().from_dict(seq.to_dict())
        norm_seq.processed_status='normalized'
        values = []
        frontground_seq = get_frontground(seq,focus_types)
        for type_ in focus_types:
            values.append(norm_seq.get_ann(type_))
        sum_values = np.array(values).sum(0)
        for type_ in focus_types:
            numerator = seq.get_ann(type_)
            denominator = sum_values
            with np.errstate(divide='ignore', invalid='ignore'):
                result = numerator / denominator
                result[denominator == 0] = 0
                result[np.logical_not(frontground_seq)] = 0
                norm_seq.set_ann(type_,result)
        for type_ in other_types:
            norm_seq.set_ann(type_,seq.get_ann(type_))
        return norm_seq

def is_full_annotated(seq, focus_types=None):
    focus_types = focus_types or seq.ANN_TYPES
    values = []
    for type_ in focus_types:
        values.append(seq.get_ann(type_))
    status = not np.any(np.array(values).sum(0)==0)
    return status

def is_value_sum_to_length(seq, focus_types=None):
    focus_types = focus_types or seq.ANN_TYPES
    values = [0]*seq.length
    for type_ in focus_types:
        values += seq.get_ann(type_)
    return sum(values)==seq.length

def is_binary(seq, focus_types=None):
    focus_types = focus_types or seq.ANN_TYPES
    values = []
    for type_ in focus_types:
        values.append(seq.get_ann(type_))
    is_binary_ = np.all(np.isin(np.array(values), [0.0,1.0]))
    return is_binary_

def is_one_hot(seq, focus_types=None):
    is_binary_ = is_binary(seq, focus_types)
    is_full_annotated_ = is_full_annotated(seq, focus_types)
    is_sum_to_length_ = is_value_sum_to_length(seq, focus_types)
    return is_binary_ and is_full_annotated_ and is_sum_to_length_

def _get_one_hot_by_max(seq, focus_types=None):
    focus_types = focus_types or seq.ANN_TYPES
    one_hot_seq = get_normalized(seq,focus_types)
    one_hot_seq.processed_status = 'one_hot'
    other_types = _get_unfocus_types(seq.ANN_TYPES,focus_types)
    values = []
    one_hot_value = {}
    for type_ in focus_types:
        values.append(one_hot_seq.get_ann(type_))
        one_hot_value[type_] = [0]*one_hot_seq.length
    one_hot_indice = np.argmax(values,0)
    for index,one_hot_index in enumerate(one_hot_indice):
        one_hot_type = focus_types[one_hot_index]
        one_hot_value[one_hot_type][index] = 1
    for type_ in focus_types:
        one_hot_seq.set_ann(type_,one_hot_value[type_])
    for type_ in other_types:
        one_hot_seq.set_ann(type_,seq.get_ann(type_))
    return one_hot_seq

def _get_one_hot_by_order(seq, focus_types=None):
    focus_types = focus_types or seq.ANN_TYPES
    one_hot_seq = get_normalized(seq,focus_types)
    one_hot_seq.processed_status='one_hot'
    other_types = _get_unfocus_types(seq.ANN_TYPES,focus_types)
    temp = AnnSequence().from_dict(one_hot_seq.to_dict())
    temp.clean_space()
    temp.ANN_TYPES = ['focused','focusing','temp']
    temp.init_space()
    for type_ in focus_types:
        temp.set_ann('focusing',one_hot_seq.get_ann(type_))
        temp.op_not_ann('temp','focusing','focused')
        temp.op_or_ann('focused','focusing','focused')
        one_hot_seq.set_ann(type_,temp.get_ann('temp'))
    for type_ in other_types:
        one_hot_seq.set_ann(type_,seq.get_ann(type_))
    return one_hot_seq

def get_one_hot(seq, focus_types=None, method='max'):
    if seq.processed_status=='one_hot':
        return seq
    else:
        focus_types = focus_types or seq.ANN_TYPES
        if not is_full_annotated(seq,focus_types):
            raise Exception("Sequence is not fully annotated")
        if method == 'max':
            return _get_one_hot_by_max(seq,focus_types)
        elif method == 'order':
            return _get_one_hot_by_order(seq,focus_types)
        else:
            raise Exception("Method ,"+str(method)+", is not supported")

def get_background(seq, frontground_types=None):
    frontground_types = frontground_types or seq.ANN_TYPES
    return  np.logical_not(get_frontground(seq,frontground_types))

def get_seq_with_added_type(ann_seq,status_dict):
    extra_types = list(status_dict.keys())
    combined_seq = AnnSequence()
    combined_seq.from_dict(ann_seq.to_dict(without_data=True))
    combined_seq.processed_status = None
    combined_seq.ANN_TYPES = ann_seq.ANN_TYPES + extra_types
    combined_seq.clean_space()
    combined_seq.init_space()
    for type_ in ann_seq.ANN_TYPES:
        combined_seq.set_ann(type_,ann_seq.get_ann(type_))
    for key,value in status_dict.items():
        combined_seq.set_ann(key,value)
    return combined_seq

def get_frontground(seq,frontground_types=None):
    frontground_types = frontground_types or seq.ANN_TYPES
    frontground_seq = np.array([0]*seq.length)
    for type_ in frontground_types:
        frontground_seq = np.logical_or(frontground_seq,seq.get_ann(type_))
    return frontground_seq

def simplify_seq(seq,replace):
    ann_seq = seq.copy()
    ann_seq.clean_space()
    ann_seq.ANN_TYPES = list(replace.keys())
    ann_seq.init_space()
    for key_ in ann_seq.ANN_TYPES:
        for type_ in replace[key_]:
            ann_seq.add_ann(key_,seq.get_ann(type_))
    return ann_seq

def class_count(ann_seq):
    ann_count = {}
    for type_ in ann_seq.ANN_TYPES:
        ann_count[type_] = np.sum(ann_seq.get_ann(type_))
    return ann_count

def seq2vecs(ann_seq,ann_types=None):
    warn = ("\n\n!!!\n"
            "\tAnnotation sequence will be rearranged from 5' to 3'.\n"
            "\tThe plus strand sequence will stay the same,"
            " but the minus strand sequence will be flipped!\n"
            "!!!\n")
    warnings.warn(warn)
    ann_types = ann_types or ann_seq.ANN_TYPES
    ann = []
    for type_ in ann_types:
        value = ann_seq.get_ann(type_)
        if ann_seq.strand == 'plus':
            ann.append(value)
        elif ann_seq.strand == 'minus':
            ann.append(np.flip(value,0))
        else:
            raise InvalidStrandType(ann_seq.strand)
    return np.transpose(ann)

def vecs2seq(vecs,id_,strand,ann_types,length=None):
    #vecs shape is channel,length
    if vecs.shape[0] != len(ann_types):
        raise Exception("The number({}) of annotation type is not match with the channel number({}).".format(len(ann_types),vecs.shape[0]))
    ann_seq = AnnSequence()
    ann_seq.ANN_TYPES = ann_types
    ann_seq.id=id_
    if length is None:
        ann_seq.length = vecs.shape[1]
    else:
        ann_seq.length = length
    ann_seq.strand = strand
    ann_seq.init_space()
    length = ann_seq.length
    for index,type_ in  enumerate(ann_types):
        ann_seq.set_ann(type_,vecs[index][:length])
    return ann_seq

def get_binary(seq):
    binary_seq = seq.copy().clean_space().init_space()
    for type_ in binary_seq.ANN_TYPES:
        temp = np.array(seq.get_ann(type_))
        temp = (temp > 0).astype(int)
        binary_seq.set_ann(type_,temp)
    return binary_seq

def get_mixed_seq(seq,map_=None):
    if 'DANGER' in seq.ANN_TYPES:
        raise Exception("Sequence {} has invalid annotation type, DANGER".format(seq.id))
    if map_ is not None and 'DANGER' not in map_.keys():
        raise Exception("Map must have key, DANGER")
    vecs = []
    for type_ in seq.ANN_TYPES:
        val = np.array(seq.get_ann(type_)) != 0.0
        vecs.append(val.astype(int))
    if map_ is None:
        map_ = {}
        max_val = 2**len(seq.ANN_TYPES)-1
        format_ = "0{}b".format(len(seq.ANN_TYPES))
        for id_ in range(0,max_val+1):
            bits = list(format(id_,format_))
            name = ""
            for bit,type_ in zip(bits,seq.ANN_TYPES):
                if bit == '1':
                    if name == "":
                        name = type_
                    else:
                        name = name + "_" + type_
            if name == "":
                name = 'DANGER'
            map_[name] = [int(bit) for bit in bits]
    t_vecs = np.array(vecs).transpose()
    mixed_type_seq = seq.copy().clean_space()
    mixed_type_seq.ANN_TYPES = list(map_.keys())
    mixed_type_seq.init_space()
    for type_,kernel in map_.items():
        ann = np.all(t_vecs==kernel,1).astype('int')
        mixed_type_seq.set_ann(type_,ann)
    valid_types = []
    for type_ in mixed_type_seq.ANN_TYPES:
        sum_ = sum(np.array(mixed_type_seq.get_ann(type_)) != 0.0)
        if sum_ > 0:
            valid_types.append(type_)

    if is_one_hot(mixed_type_seq):
        mixed_type_seq.processed_status = 'one_hot'
    else:
        raise Exception("Sequence {} cannot convert to one-hot sequence".format(seq.id))
    return mixed_type_seq

def get_start(signal):
    return [m.start()+1 for m in re.finditer('01', signal)]

def get_end(signal):
    return [m.start()+1 for m in re.finditer('10', signal)]

def is_valid_strand(seq):
    return seq.strand in ['plus','minus']

def get_tss(seq):
    if not is_valid_strand(seq):
        raise InvalidStrandType(seq.strand)
    signal = ''.join(seq.get_ann('gene').astype(int).astype(str))
    if seq.strand == 'plus':
        return get_start(signal)
    else:
        return get_end(signal)

def get_ca(seq):
    if not is_valid_strand(seq):
        raise InvalidStrandType(seq.strand)
    signal = ''.join(seq.get_ann('gene').astype(int).astype(str))
    if seq.strand == 'plus':
        return get_end(signal)
    else:
        return get_start(signal)

def get_donor(seq):
    if not is_valid_strand(seq):
        raise InvalidStrandType(seq.strand)
    signal = ''.join(seq.get_ann('intron').astype(int).astype(str))
    if seq.strand == 'plus':
        return get_start(signal)
    else:
        return get_end(signal)

def get_accept(seq):
    if not is_valid_strand(seq):
        raise InvalidStrandType(seq.strand)
    signal = ''.join(seq.get_ann('intron').astype(int).astype(str))
    if seq.strand == 'plus':
        return get_end(signal)
    else:
        return get_start(signal)

def get_seq_with_site_ann(seq,gene_map,set_non_site=False):
    simplified = simplify_seq(seq,gene_map)
    TSSs = get_tss(simplified)
    CAs = get_ca(simplified)
    DSs = get_donor(seq)
    ASs = get_accept(seq)
    site_ann = {}
    site_names = ['TSSs','ca_sites','donor_sites','accept_sites']
    multiple_sites = [TSSs,CAs,DSs,ASs]
    non_site_ann = np.ones(seq.length)
    for name,sites in zip(site_names,multiple_sites):
        template = np.zeros(seq.length)
        for site in sites:
            template[site] = 1
            non_site_ann[site] = 0
        site_ann[name] = template    

    if set_non_site:
        site_ann['non_site'] = non_site_ann

    seq = get_seq_with_added_type(seq,site_ann)
    return seq
