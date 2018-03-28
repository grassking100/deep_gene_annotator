"""This submodule will handler character sequence and one-hot encoding seqeunce conversion"""
from . import CodeException, SeqException
import numpy as np
class SeqConverter():
    def __init__(self,code_vec_dictionay=None,codes=None,is_case_sensitivity=False):
        if code_vec_dictionay is not None:
            if self._is_dict_inversible(code_vec_dictionay):
                if is_case_sensitivity:
                    self._code_vec_dictionay = code_vec_dictionay
                    self._codes = list(code_vec_dictionay.keys())
                else:
                    self._code_vec_dictionay = {}
                    self._codes = []
                    for code,vec in code_vec_dictionay.items():
                        code_key = str(code).upper()
                        self._code_vec_dictionay[code_key]=vec
                        self._codes.append(code_key)
            else:
                raise Exception("Diciotnary is not inversible")
        else:
            self._codes=codes or self._default_code()
            if self._is_values_unique(self._codes):
                self._code_vec_dictionary = self._codes_to_vecs_dictionay(self._codes)
            else:
                raise Exception("Codes is not unique")
        self._vec_code_dictionary = {str(v):k for k,v in self._code_vec_dictionary.items()}
        self._is_case_sensitivity = is_case_sensitivity
    def _is_values_unique(self, values):
        return len(values)==len(set(values))
    def _is_dict_inversible(self, dict_):
        dict_values = list(dict_.values())
        return self._is_values_unique(dict_values)
    def _default_code(self):
        return ['A', 'T', 'C', 'G']
    def _codes_to_vecs_dictionay(self, codes):
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
        """convert element by dictionay"""
        result = dictionary.get(self._preprocess(element))
        if result is None:
            raise CodeException(str(element),list(dictionary.keys()))
        else:
            return result
    def _array_convert(self,array,element_convert_method):
        """convert sequence by dictionay"""
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
        return self._array_convert(seq,self.code2vec)
    def vecs2seq(self,vecs,join=True):
        """convert vector of vectir to converted result"""
        result = self._array_convert(vecs,self.vec2code)
        if join:
            return "".join(result)
        else:
            return result
