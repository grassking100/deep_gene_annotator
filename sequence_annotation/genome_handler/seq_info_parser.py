import pandas as pd
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from . import NegativeNumberException,InvalidStrandType,NotPositiveException
class SeqInfoParser(metaclass=ABCMeta):
    def parse(self,data_list):
        returned_list = []
        for data in data_list:
            returned_list.append(self._parse(data))
        return returned_list
    @abstractmethod
    def _parse(self):
        pass
    @abstractmethod
    def _validate(self):
        pass

class UscuInfoParser(SeqInfoParser):
    """Purpose:Parse file from USCU table and get data which stored infomration(zero-based)"""
    def _parse(self,data):
        value_int_zero_based = ['txStart','cdsStart','exonCount']
        value_int_one_based = ['txEnd','cdsEnd']
        value_str = ['chrom','strand','name']
        self._result = []
        temp_data = {}
        for key in value_int_zero_based:
            temp_data[key] = int(data[key])
        for key in value_int_one_based:
            temp_data[key] = int(data[key])-1
        for key in value_str:
            temp_data[key] = str(data[key])
        temp_data['exonStarts'] = np.array(str(data['exonStarts'][:-1]).split(","),dtype="int")
        temp_data['exonEnds'] = np.array(str(data['exonEnds'][:-1]).split(","),dtype="int")-1
        if data['exonCount'] <= 0:
            raise NotPositiveException("exonCount",data['exonCount'])
        strand = temp_data['strand']
        if strand=='+':
            temp_data['strand'] = "plus"
        elif strand=='-':
            temp_data['strand'] = "minus"
        else:
            raise InvalidStrandType(strand)
        self._validate(temp_data)
        return temp_data

    def _validate(self,data):
        value_int = ['txStart','txEnd','cdsStart','cdsEnd','exonCount']
        value_int_list = ['exonEnds','exonStarts']
        for key in value_int:
            if data[key] < 0:
                raise NegativeNumberException(key,data[key])
        for key in value_int_list:
            if data[key] is not None and np.any(data[key] < 0):
                raise NegativeNumberException(key,data[key])

class EnsemblInfoParser(SeqInfoParser):
    """Purpose:Parse file from Ensembl table and get data which stored infomration(zero-based)"""
    def _parse(self,data):
        temp_data = {}
        temp_data['tx_start'] = int(data['Transcript start (bp)']) -1
        temp_data['tx_end'] = int(data['Transcript end (bp)']) -1
        """Convert type"""
        convert_name = {'Exon region start (bp)':'exons_start',
                        'Exon region end (bp)':'exons_end',
                        'Genomic coding start':'cdss_start',
                        'Genomic coding end':'cdss_end',
                        '5\' UTR start':'utrs_5_start',
                        '3\' UTR start':'utrs_3_start',
                        '5\' UTR end':'utrs_5_end',
                        '3\' UTR end':'utrs_3_end'}
        for origin,new_name in convert_name.items():
            temp = np.array(data[origin].split(','))
            temp_list=[]
            for value in temp:
                if value != "Null":
                    temp_list.append(int(float(value))-1)
            temp_data[new_name]=np.array(sorted(temp_list))
        """Other"""
        temp_data['protein_id'] = data['Protein stable ID']
        strand = str(data['Strand'])
        if strand=='1':
            temp_data['strand'] = "plus"
        elif strand=='-1':
            temp_data['strand'] = "minus"
        else:
            raise InvalidStrandType(strand)
        temp_data['chrom'] = data['Chromosome/scaffold name']
        self._validate(temp_data)
        return temp_data
    def _validate(self,data):
        value_int = ['tx_start','tx_end']
        value_int_list = ['cdss_start','cdss_end','utrs_5_start',
                          'utrs_3_start','utrs_5_end','utrs_3_end']
        for key in value_int:
            if data[key] < 0:
                raise NegativeNumberException(key,data[key])
        for key in value_int_list:
            if data[key] is not None and np.any(data[key] < 0):
                raise NegativeNumberException(key,data[key])