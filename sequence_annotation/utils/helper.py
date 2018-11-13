import os
import errno
def create_folder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as erro:
        if erro.errno != errno.EEXIST:
            raise

def get_protected_attrs_names(object_):
    class_name = object_.__class__.__name__
    attrs = [attr for attr in dir(object_) if attr.startswith('_') 
             and not attr.endswith('__')
             and not attr.startswith('_'+class_name+'__')]
    return attrs

def reverse_weights(cls, class_counts, epsilon=1):
    scale = len(class_counts.keys())
    raw_weights = {}
    weights = {}
    for type_,count in class_counts.items():
        if count > 0:
            weight = 1 / count
        else:
            if epsilon > 0:
                weight = 1 / (count+epsilon)
            else:
                raise Exception(type_+" has zero count,so it cannot get reversed count weight")
        raw_weights[type_] = weight
    sum_raw_weights = sum(raw_weights.values())
    for type_,weight in raw_weights.items():
        weights[type_] = scale * weight / (sum_raw_weights)
    return weights
