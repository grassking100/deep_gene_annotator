class NotBinaryException(Exception):
    def __init__(self, seq_id):
        msg = "Sequence, {}, is not binary sequence".format(seq_id)
        super().__init__(msg)


class NotOneHotException(Exception):
    def __init__(self, seq_id):
        msg = "Sequence, {}, is not one hot encoded".format(seq_id)
        super().__init__(msg)


class ProcessedStatusNotSatisfied(Exception):
    def __init__(self, get_status, predict_status):
        msg = "Get {}, but it is expect to be {}".format(
            get_status, predict_status)
        super().__init__(msg)


class InvalidAnnotation(Exception):
    def __init__(self, ann_type=None, valid_types=None):
        type_ = ""
        if ann_type is not None:
            type_ = "," + str(ann_type) + ","
        msg = "Annotation type{} is not expected".format(type_)
        if valid_types is not None:
            msg += ", valid type are {}".format(valid_types)
        super().__init__(msg)


class NotSameSizeException(Exception):
    def __init__(self, lhs, rhs):
        msg = "{} and {} should have same size".format(lhs, rhs)
        super().__init__(msg)
