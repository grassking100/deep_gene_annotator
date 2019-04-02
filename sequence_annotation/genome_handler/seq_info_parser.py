from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import ast
from ..utils.exception import NegativeNumberException,InvalidStrandType,NotPositiveException
class SeqInfoParser(metaclass=ABCMeta):
    def parse(self,data_list):
        returned_list = []
        for data in data_list:
            returned_list.append(self._parse(data))
        return returned_list
    @abstractmethod
    def _parse(self,data):
        pass
    @abstractmethod
    def _validate(self,data):
        pass

class BedInfoParser(SeqInfoParser):
    """
       Purpose:Parse file from BED file and get data which stored infomration(zero-based)
       See format:https://genome.ucsc.edu/FAQ/FAQformat.html#format1
    """
    def __init__(self,bed_type="bed_12"):
        if bed_type in ["bed_12","bed_6"]:
            self.bed_type = bed_type
        else:
            raise Exception("Wrong bed type")
    def _parse(self,data):
        new_data = {}
        value_int_zero_based = ['chromStart']
        value_int_one_based = ['chromEnd']
        columns = ['chrom','chromStart','chromEnd','name','score',
                   'strand','thickStart','thickEnd','itemRgb',
                   'blockCount','blockSizes','blockStarts']
        for index, column in enumerate(columns[:len(data)]):
            new_data[column] = data[index]
        for key in new_data.keys():
            new_data[key] = str(new_data[key])
        if  self.bed_type == 'bed_12':
            value_int_zero_based += ['thickStart','blockCount']
            value_int_one_based += ['thickEnd']
        for key in value_int_zero_based:
            new_data[key] = int(new_data[key])
        for key in value_int_one_based:
            new_data[key] = int(new_data[key]) - 1
        strand = new_data['strand']
        if strand == '+':
            new_data['strand'] = "plus"
        elif strand == '-':
            new_data['strand'] = "minus"
        else:
            raise InvalidStrandType(strand)
        if  self.bed_type == 'bed_12':
            block_count = new_data['blockCount']
            if block_count <= 0:
                raise NotPositiveException("blockCount",block_count)
            for key in ['blockStarts','blockSizes']:
                list_ = new_data[key].split(",")[:block_count]
                new_data[key] = np.array(list_,dtype="int")
        self._validate(new_data)
        return new_data

    def _validate(self,data):
        value_int = ['chromStart','chromEnd']
        if  self.bed_type == 'bed_12':
            value_int += ['thickStart','blockCount','thickEnd']
            value_int_list = ['blockStarts','blockSizes']
            for key in value_int_list:
                if np.any(data[key] < 0):
                    raise NegativeNumberException(key,data[key])
        for key in value_int:
            if data[key] < 0:
                raise NegativeNumberException(key,data[key])


class UCSCInfoParser(SeqInfoParser):
    """Purpose:Parse file from UCSC table and get data which stored infomration(zero-based)"""
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
    """Purpose:Parse file from Ensembl table and get data which stored infomration(zero-based)"""
    def _parse(self,data):
        temp_data = {}
        temp_data['tx_start'] = int(data['Transcript start (bp)']) - 1
        temp_data['tx_end'] = int(data['Transcript end (bp)']) - 1
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
            val_list = ast.literal_eval(str(data[origin]))
            if not isinstance(val_list,list):
                val_list = [val_list]
            temp_list=[int(val)-1 for val in val_list]
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
                          'utrs_3_start','utrs_5_end','utrs_3_end',
                          'exons_start','exons_end']
        for key in value_int:
            if data[key] < 0:
                raise NegativeNumberException(key,data[key])
        for key in value_int_list:
            if data[key] is not None and np.any(data[key] < 0):
                raise NegativeNumberException(key,data[key])