from abc import ABCMeta
from abc import abstractmethod
import ast
import numpy as np
import pandas as pd
from ..utils.exception import NegativeNumberException,NotPositiveException
from ..utils.utils import BED_COLUMNS
from .exception import InvalidStrandType

class SeqInfoParser(metaclass=ABCMeta):
    """Parse file from list and get data which stored infomration(zero-based)"""
    def parse(self,path):
        data_list = self._load_data(path)
        returned_list = []
        for data in data_list:
            returned_list.append(self._parse(data))
        return returned_list

    @abstractmethod
    def _load_data(self,path):
        pass
    
    @abstractmethod
    def _parse(self,data):
        pass

    @abstractmethod
    def _validate(self,data):
        pass

class BedInfoParser(SeqInfoParser):
    """Parse file from BED file and get data which stored infomration(zero-based)
       See format:https://genome.ucsc.edu/FAQ/FAQformat.html#format1
    """
    def __init__(self,bed_type="bed_12"):
        if bed_type in ["bed_12","bed_6"]:
            self.bed_type = bed_type
        else:
            raise Exception("Wrong bed type")

    def _load_data(self,path):
        df = pd.read_csv(path,header=None,sep='\t')
        return df.to_dict('record')
            
    def _parse(self,data):
        parsed_data = {}
        value_int_zero_based = ['start']
        value_int_one_based = ['end']
        for index, column in enumerate(BED_COLUMNS[:len(data)]):
            parsed_data[column] = data[index]
        for key,value in parsed_data.items():
            parsed_data[key] = str(value)
        if  self.bed_type == 'bed_12':
            value_int_zero_based += ['thick_start','count']
            value_int_one_based += ['thick_end']
        for key in value_int_zero_based:
            parsed_data[key] = int(parsed_data[key])
        for key in value_int_one_based:
            parsed_data[key] = int(parsed_data[key]) - 1
        strand = parsed_data['strand']
        if strand == '+':
            parsed_data['strand'] = "plus"
        elif strand == '-':
            parsed_data['strand'] = "minus"
        else:
            raise InvalidStrandType(strand)
        if  self.bed_type == 'bed_12':
            for key in ['block_related_start','block_size']:
                list_ = parsed_data[key].split(",")
                parsed_data[key] = [int(item) for item in list_]
            starts = parsed_data['block_related_start']
            sizes = parsed_data['block_size']
            parsed_data['block_related_end'] = [start+size-1 for start, size in zip(starts,sizes)]
        self._validate(parsed_data)
        return parsed_data

    def _validate(self,data):
        value_int = ['start','end']
        for key in value_int:
            if data[key] < 0:
                raise NegativeNumberException(key,data[key])
        if  self.bed_type == 'bed_12':
            value_int += ['thick_start','count','thick_end']
            value_int_list = ['block_related_start','block_size']
            for key in value_int_list:
                if np.any(np.array(data[key]) < 0):
                    raise NegativeNumberException(key,data[key])
            count = data['count']
            if count <= 0:
                raise NotPositiveException("count",count)
            if len(data['block_related_start']) != count or len(data['block_size']) != count:
                raise Exception("Start sites or sizes number are not same as count")

class UCSCInfoParser(SeqInfoParser):
    """Parse file from UCSC table and get data which stored infomration(zero-based)"""
    def _load_data(self,path):
        df = pd.read_csv(path,sep='\t')
        return df.to_dict('record')
    
    def _parse(self,data):
        value_int_zero_based = ['txStart','cdsStart','exonCount']
        value_int_one_based = ['txEnd','cdsEnd']
        value_str = ['chrom','strand','name']
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
    """Parse file from Ensembl table and get data which stored infomration(zero-based)"""
    def _load_data(self,path):
        df = pd.read_csv(path,sep='\t')
        return df.to_dict('record')
    
    def _parse(self,data):
        temp_data = {}
        temp_data['tx_start'] = int(data['Transcript start (bp)']) - 1
        temp_data['tx_end'] = int(data['Transcript end (bp)']) - 1
        #Convert type
        convert_name = {'Exon region start (bp)':'exons_start',
                        'Exon region end (bp)':'exons_end',
                        'Genomic coding start':'cdss_start',
                        'Genomic coding end':'cdss_end',
                        '5\' UTR start':'utrs_5_start',
                        '3\' UTR start':'utrs_3_start',
                        '5\' UTR end':'utrs_5_end',
                        '3\' UTR end':'utrs_3_end'}
        for origin,new_name in convert_name.items():
            val_list = ast.literal_eval(str(data[origin]))
            if not isinstance(val_list,list):
                val_list = [val_list]
            temp_list=[int(val)-1 for val in val_list]
            temp_data[new_name]=np.array(sorted(temp_list))
        #Other
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
                          'utrs_3_start','utrs_5_end','utrs_3_end',
                          'exons_start','exons_end']
        for key in value_int:
            if data[key] < 0:
                raise NegativeNumberException(key,data[key])
        for key in value_int_list:
            if data[key] is not None and np.any(data[key] < 0):
                raise NegativeNumberException(key,data[key])
