"""This submodule will handler character sequence and one-hot encoding seqeunce conversion"""
from ..utils.exception import CodeException, SeqException
import numpy as np

DNA_CODES = 'ATCG'
AA_CODES = 'ARNDCQEGHILKMFPSTWYV'

class SeqConverter:
    def __init__(self,code_vec_dictionary=None,codes=None,
                 is_case_sensitivity=False,with_soft_masked_status=False):
        if code_vec_dictionary is not None:
            if self._is_dict_inversible(code_vec_dictionary):
                if is_case_sensitivity:
                    self._code_vec_dictionary = code_vec_dictionary
                    self._codes = list(code_vec_dictionary.keys())
                else:
                    self._code_vec_dictionary = {}
                    self._codes = []
                    for code,vec in code_vec_dictionary.items():
                        code_key = str(code).upper()
                        self._code_vec_dictionary[code_key]=vec
                        self._codes.append(code_key)
            else:
                raise Exception("Diciotnary is not inversible")
        else:
            self._codes=codes or list(DNA_CODES)
            if self._is_values_unique(self._codes):
                self._code_vec_dictionary = self._codes_to_vecs_dictionary(self._codes)
            else:
                raise Exception("Codes is not unique")
        self._vec_code_dictionary = {str(v):k for k,v in self._code_vec_dictionary.items()}
        self._is_case_sensitivity = is_case_sensitivity
        self._with_soft_masked_status = with_soft_masked_status
    def _add_soft_masked(self,seq,vecs):
        soft_mask = self._soft_masked_status(seq)
        vecs = np.array(vecs)
        num,length = vecs.shape
        temp = np.zeros((num,length+1))
        temp[:num,:length] = temp[:num,:length] + vecs
        temp.T[length] = soft_mask
        return temp
    def _soft_masked_status(self,seq):
        soft_masked_status = [char.islower() for char in seq]
        return soft_masked_status
    def _is_values_unique(self, values):
        return len(values)==len(set(values))
    def _is_dict_inversible(self, dict_):
        dict_values = list(dict_.values())
        return self._is_values_unique(dict_values)
    def _codes_to_vecs_dictionary(self, codes):
        dict_ = {}
        for index,code in enumerate(codes):
            zero = np.zeros(len(codes),dtype='int')
            zero[index] = 1
            dict_[code]=zero.tolist()
        return dict_
    def _preprocess(self,value):
        value =str(value)
        if self._is_case_sensitivity:
            return value
        else:
            return value.upper()
    def _element_convert(self,element,dictionary):
        """convert element by dictionary"""
        result = dictionary.get(self._preprocess(element))
        if result is None:
            raise CodeException(str(element),list(dictionary.keys()))
        else:
            return result
    def _array_convert(self,array,element_convert_method):
        """convert sequence by dictionaty"""
        code_list = list(array)
        arr = []
        for code in code_list:
            try:
                arr.append(element_convert_method(code))
            except CodeException as exp:
                raise SeqException(exp.invalid_code,exp.valid_codes)
        return arr
    def code2vec(self, code):
        """convert DNA code to one hot encoding"""
        return self._element_convert(code,self._code_vec_dictionary)
    def vec2code(self, vec):
        """convert DNA code to one hot encoding"""
        return self._element_convert(vec,self._vec_code_dictionary)
    def seq2vecs(self,seq):
        """convert sequence to one hot encoding sequence"""
        vecs = self._array_convert(seq,self.code2vec)
        if self._with_soft_masked_status:
            vecs = self._add_soft_masked(seq,vecs)
        return vecs
    def vecs2seq(self,vecs,join=True):
        """convert vector of vectir to converted result"""
        if self._with_soft_masked_status:
            vecs = np.array(vecs,dtype='int')
            num,length = vecs.shape
            seq = self._array_convert(vecs[:num,:length-1].tolist(),self.vec2code)
            seq = self._mark_soft_masked(seq,vecs)
        else:
            seq = self._array_convert(vecs,self.vec2code)
        if join:
            return "".join(seq)
        else:
            return seq
    def seqs2dict_vec(self,seqs,discard_invalid_seq=False):
        """convert dictionary of seqeucnces to dictionary of one-hot encoding vectors"""
        data = {}
        for name,seq in seqs.items():
            try:
                data[name] = self.seq2vecs(seq)
            except SeqException as exp:
                if not discard_invalid_seq:
                    raise exp
        return data

