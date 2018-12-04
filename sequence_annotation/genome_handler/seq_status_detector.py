from abc import ABCMeta

class SeqStatusDetector(metaclass=ABCMeta):
    def get_invalid_status(self,seq,invalid_chars=None):
        invalid_chars = invalid_chars or []
        invalid_chars = [char.lower() for char in invalid_chars]
        #Get its invalid character status
        invalid_status = [char.lower() in invalid_chars for char in seq]
        return invalid_status
    def get_softed_masked_status(self,seq):
        #Get its soft masked status
        soft_masked_status = [char.islower() for char in seq]
        return soft_masked_status
