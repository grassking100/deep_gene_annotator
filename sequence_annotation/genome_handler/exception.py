class NotBinaryException(Exception):
    def __init__(self,seq_id):
        msg = "Sequence, {}, is not binary sequence".format(seq_id)
        super().__init__(msg)

class NotOneHotException(Exception):
    def __init__(self,seq_id):
        msg = "Sequence, {}, is not one hot encoded".format(seq_id)
        super().__init__(msg)

class InvalidStrandType(Exception):
    def __init__(self, strand_type= None):
        type_=""
        if strand_type is not None:
            type_ = ","+ str(strand_type)+","
        msg = ("Strand type"+str(type_)+" "
               "is not expected")
        super().__init__(msg)

class ProcessedStatusNotSatisfied(Exception):
    def __init__(self,get_status,predict_status):
        msg = ("Get "+str(get_status)+""
               ",but it is expect to be "
               ""+str(predict_status))
        super().__init__(msg)

class InvalidAnnotation(Exception):
    def __init__(self, ann_type = None):
        type_=""
        if ann_type is not None:
            type_ = ","+str(ann_type)+","
        msg = ("Annotation type"+str(type_)+" "
               "is not expected")
        super().__init__(msg)
