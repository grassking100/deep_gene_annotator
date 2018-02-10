import pandas as pd
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from . import AttrValidator
from . import validate_return
from . import NegativeNumberException,InvalidStrandType,ReturnNoneException
class SeqInfoParser(metaclass=ABCMeta):
    def __init__(self,file_path):
        self._result = None
        self._raw_data = None
        self._read_file(file_path)
    @abstractmethod
    def _read_file(self, file):
        pass
    @abstractmethod
    def parse(self):
        pass
    @abstractmethod
    def _validate(self):
        pass
    @property
    @validate_return("use method parse before access the data")
    def result(self):
        return self._result
class USCUParser(SeqInfoParser):
    """Purpose:Parse file from USCU table and get data which stored infomration(zero-based)"""
    def _validate(self):
        attr_validator = AttrValidator(self)
        attr_validator.validate_attr_names = ['_raw_data']
        attr_validator.validate()
    def _read_file(self, file):
        self._raw_data = pd.read_csv(file, sep='\t')
    def _validate_data(self):
        value_int = ['txStart','txEnd','cdsStart','cdsEnd','exonCount']
        value_int_list = ['exonEnds','exonStarts']
        for row in self._result:
            if not (row['strand']=='+' or row['strand']=='-'):
                raise InvalidStrandType(row['strand'])
            for key in value_int:
                if row[key] < 0:
                    raise NegativeNumberException(key,row[key])
            for key in value_int_list:
                if row[key] is not None and np.any(row[key] < 0):
                    raise NegativeNumberException(key,row[key])
    def parse(self):
        self._validate()
        value_int_zero_based_list = ['txStart','cdsStart','exonCount']
        value_int_one_based_list = ['txEnd','cdsEnd']
        value_str_list = ['chrom','strand','name']
        self._result = []
        temp_data = {}
        for key in value_int_zero_based_list:
            temp_data[key] = np.array(self._raw_data[key].tolist(),dtype="int")
        for key in value_int_one_based_list:
            temp_data[key] = np.array(self._raw_data[key].tolist(),dtype="int")-1
        for key in value_str_list:
            temp_data[key] = np.array(self._raw_data[key].tolist(),dtype="str")
        end_indice = []
        start_indice = []
        ends = np.array(self._raw_data['exonEnds'].tolist())
        starts = np.array(self._raw_data['exonStarts'].tolist())
        for start, end, count in zip(starts, ends,temp_data['exonCount']):
            if count==0:
                start_indice += [None]
                end_indice += [None]
            else:
                start_indice += [np.array(start[:-1].split(","),dtype="int")]
                end_indice += [(np.array(end[:-1].split(","),dtype="int")-1)]
        temp_data['exonEnds'] =  np.array(end_indice)
        temp_data['exonStarts'] = np.array(start_indice)
        for index in self._raw_data.index:
            row = {}
            for key in temp_data.keys():
                row[key] = temp_data[key][index]
            self._result.append(row)
        self._validate_data()
        