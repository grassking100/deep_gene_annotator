from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from ..utils.exception import NegativeNumberException, NotPositiveException
from .utils import read_bed, PLUS,MINUS, InvalidStrandType

class SeqInfoParser(metaclass=ABCMeta):
    """Parse file from list and get data which stored infomration(zero-nt based)"""

    def parse(self, path_or_df):
        data_list = self._load_data(path_or_df)
        returned_list = []
        for data in data_list:
            returned_list.append(self._parse(data))
        return returned_list

    @abstractmethod
    def _load_data(self, path_or_df):
        pass

    @abstractmethod
    def _parse(self, data):
        pass

    @abstractmethod
    def _validate(self, data):
        pass


class BedInfoParser(SeqInfoParser):
    """Parse file from BED file and get data which stored infomration(zero-base based)
       See format:https://genome.ucsc.edu/FAQ/FAQformat.html#format1
    """

    def _load_data(self, path_or_df):
        if isinstance(path_or_df, str):
            bed = read_bed(path_or_df)
        else:
            bed = path_or_df.copy()
        # Convert one-base based data to zero-base based
        bed['start'] = bed['start'] - 1
        bed['end'] = bed['end'] - 1
        bed['thick_start'] = bed['thick_start'] - 1
        bed['thick_end'] = bed['thick_end'] - 1
        return bed.to_dict('record')

    def _parse(self, data):
        parsed_data = dict(data)
        strand = parsed_data['strand']
        if strand == '+':
            parsed_data['strand'] = PLUS
        elif strand == '-':
            parsed_data['strand'] = MINUS
        else:
            raise InvalidStrandType(strand)
        for key in ['block_related_start', 'block_size']:
            list_ = parsed_data[key].split(",")
            parsed_data[key] = [int(item) for item in list_]
        starts = parsed_data['block_related_start']
        sizes = parsed_data['block_size']
        parsed_data['block_related_end'] = [start + size - 1 for start, size in zip(starts, sizes)]
        del parsed_data['block_size']
        self._validate(parsed_data)
        return parsed_data

    def _validate(self, data):
        transcript_size = data['end'] - data['start'] + 1
        value_int = ['start', 'end','thick_start', 'count', 'thick_end']
        for key in value_int:
            if data[key] < 0:
                raise NegativeNumberException(key, data[key])
                
        for rel_start,rel_end in zip(data['block_related_start'],data['block_related_end']):
            if rel_start < 0:
                raise NegativeNumberException('block relative start', rel_start)
            if rel_end < 0:
                raise NegativeNumberException('block relative start', rel_start)
            if rel_end >= transcript_size:
                raise Exception("Wrong relative end")
                
            if rel_end - rel_start + 1 > transcript_size:
                print(rel_end,rel_start)
                raise Exception("Wrong size")

        if len(data['block_related_start']) != data['count']:
            raise Exception("Start sites are not same as count")

        if len(data['block_related_end']) != data['count']:
            raise Exception("End sites are not same as count")
            
