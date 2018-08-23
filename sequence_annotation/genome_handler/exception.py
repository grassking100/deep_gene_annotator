class NotOneHotException(Exception):
    def __init__(self,seq_id):
        msg = "Sequence,"+str(seq_id)+",is not one hot encoded"
        super().__init__(msg)