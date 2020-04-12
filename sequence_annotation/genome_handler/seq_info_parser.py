from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from ..utils.exception import NegativeNumberException, NotPositiveException, InvalidStrandType
from ..utils.utils import read_bed
from .sequence import PLUS, MINUS


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
            # If it is bed data which is read by read_bed
            bed = path_or_df
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
        parsed_data['block_related_end'] = [
            start + size - 1 for start, size in zip(starts, sizes)]
        self._validate(parsed_data)
        return parsed_data

    def _validate(self, data):
        value_int = ['start', 'end']
        for key in value_int:
            if data[key] < 0:
                raise NegativeNumberException(key, data[key])
        value_int += ['thick_start', 'count', 'thick_end']
        value_int_list = ['block_related_start', 'block_size']
        for key in value_int_list:
            if np.any(np.array(data[key]) < 0):
                raise NegativeNumberException(key, data[key])
        count = data['count']
        if count <= 0:
            raise NotPositiveException("count", count)
        if len(data['block_related_start']) != count or len(
                data['block_size']) != count:
            raise Exception(
                "Start sites or sizes number are not same as count")
