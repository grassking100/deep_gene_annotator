"""This submodule will handler character sequence and one-hot encoding seqeunce conversion"""
import numpy as np
from multiprocessing import Pool,cpu_count

DNA_CODES = 'ATCG'
AA_CODES = 'ARNDCQEGHILKMFPSTWYV'

class CodeException(Exception):
    """Raising when input code is not in in defined space"""
    def __init__(self, invalid_code, valid_codes=None):
        self._invalid_code = invalid_code
        self._valid_codes = valid_codes
        mess = str(invalid_code) + ' is not in defined space'
        if self._valid_codes is not None:
            mess += (",valid codes are " + str(self._valid_codes))
        super().__init__(mess)

    @property
    def invalid_code(self):
        return self._invalid_code

    @property
    def valid_codes(self):
        return self._valid_codes


class SeqException(Exception):
    """Raising when input sequences has at least a code is not in in defined space"""
    def __init__(self, invalid_code, valid_codes=None):
        mess = "Seqeunce has a invalid code," + str(invalid_code)
        if valid_codes is not None:
            mess += (" ,valid codes are " + str(valid_codes))
        super().__init__(mess)


class SeqConverter:
    def __init__(self,code_vec_dictionary=None,to_numpy=False,
                 codes=None,is_case_sensitivity=False):
        if code_vec_dictionary is not None:
            if self._is_dict_inversible(code_vec_dictionary):
                if is_case_sensitivity:
                    self._code_vec_dictionary = code_vec_dictionary
                    self._codes = list(code_vec_dictionary.keys())
                else:
                    self._code_vec_dictionary = {}
                    self._codes = []
                    for code, vec in code_vec_dictionary.items():
                        code_key = str(code).upper()
                        self._code_vec_dictionary[code_key] = vec
                        self._codes.append(code_key)
            else:
                raise Exception("Diciotnary is not inversible")
        else:
            self._codes = codes or list(DNA_CODES)
            if self._is_values_unique(self._codes):
                self._code_vec_dictionary = self._codes_to_vecs_dictionary(
                    self._codes)
            else:
                raise Exception("Codes is not unique")
        self._vec_code_dictionary = {
            str(v): k
            for k, v in self._code_vec_dictionary.items()
        }
        self._is_case_sensitivity = is_case_sensitivity
        self._to_numpy = to_numpy

    def _add_soft_masked(self, seq, vecs):
        soft_mask = self._soft_masked_status(seq)
        vecs = np.array(vecs)
        num, length = vecs.shape
        temp = np.zeros((num, length + 1))
        temp[:num, :length] = temp[:num, :length] + vecs
        temp.T[length] = soft_mask
        return temp

    def _soft_masked_status(self, seq):
        soft_masked_status = [char.islower() for char in seq]
        return soft_masked_status

    def _is_values_unique(self, values):
        return len(values) == len(set(values))

    def _is_dict_inversible(self, dict_):
        dict_values = list(dict_.values())
        return self._is_values_unique(dict_values)

    def _codes_to_vecs_dictionary(self, codes):
        dict_ = {}
        for index, code in enumerate(codes):
            zero = np.zeros(len(codes), dtype='int')
            zero[index] = 1
            dict_[code] = zero.tolist()
        return dict_

    def _preprocess(self, value):
        if not self._is_case_sensitivity:
            if len(value)!=1:
                value = list(''.join(value).upper())
            else:
                value = value.upper()
        return value

    def _convert_element(self, element, dictionary):
        """convert element by dictionary"""
        result = dictionary.get(str(element))
        if result is None:
            raise CodeException(str(element), list(dictionary.keys()))
        else:
            return result

    def _convert_array(self, array, element_convert_method):
        """convert sequence by dictionary"""
        code_list = list(array)
        arr = []
        for code in code_list:
            try:
                arr.append(element_convert_method(code))
            except CodeException as exp:
                raise SeqException(exp.invalid_code, exp.valid_codes)
        return arr

    def _code2vec(self, code):
        """convert DNA code to one hot encoding"""
        vec = self._convert_element(code, self._code_vec_dictionary)
        return vec
    
    def code2vec(self, code):
        """convert DNA code to one hot encoding"""
        code = self._preprocess(code)
        vec = self._code2vec(code)
        return vec

    def vec2code(self, vec):
        """convert DNA code to one hot encoding"""
        return self._convert_element(vec, self._vec_code_dictionary)

    def seq2vecs(self, seq):
        """convert sequence to one hot encoding sequence"""
        seq = self._preprocess(seq)
        vecs = self._convert_array(seq, self._code2vec)
        if self._to_numpy:
            vecs = np.array(vecs)
        return vecs

    def vecs2seq(self, vecs, join=True):
        """convert vector of vectir to converted result"""
        seq = self._convert_array(vecs, self.vec2code)
        if join:
            return "".join(seq)
        else:
            return seq

    def _seq2dict_vec(self,name,seq):
        vec = self.seq2vecs(seq)
        data = {name:vec}
        return data
        
    def seqs2dict_vec(self, seqs):
        """convert dictionary of seqeucnces to dictionary of one-hot encoding vectors"""
        multiprocess = cpu_count()
        kwarg_list = []
        for name, seq in seqs.items():
            kwarg_list.append((name,seq))
        with Pool(processes=multiprocess) as pool:
            items = pool.starmap(self._seq2dict_vec, kwarg_list)
        data = {}
        for item in items:
            data.update(item)
        return data
