import os
import sys
sys.path.append(os.path.dirname(__file__)+"/..")
from ..utils.utils import read_json,get_file_name
from ..postprocess.utils import get_data_names

class PathHelper:
    def __init__(self,raw_data_root,processed_root,trained_id=None,usage=None):
        self._raw_data_root = raw_data_root
        self._processed_root = processed_root
        self._split_root = os.path.join(raw_data_root,'split/single_strand_data/split_with_strand')
        train_val_test_path=read_json(os.path.join(self._split_root,'train_val_test_path.json'))
        self._data_usage = get_data_names(self._split_root)
        self._train_val_name = get_file_name(train_val_test_path['train_val_path'])
        self._test_name = get_file_name(train_val_test_path['test_path'])

        if trained_id is None:
            trained_id = self._train_val_name

        
        if usage is not None:
            self._file_name = self._data_usage[trained_id][usage]
        else:
            self._file_name = trained_id 
            
        self._main_kwargs_path = os.path.join(self._raw_data_root,'main_kwargs.csv')
      
        self._region_table_path = os.path.join(self._split_root,'region_table',
                                               "{}_part_region_table.tsv".format(self._file_name))

        self._fasta_path = os.path.join(self._split_root,'fasta',self._file_name+".fasta")

    
        self._answer_path = os.path.join(self._split_root,'gff',self._file_name+"_canonical_double_strand.gff3")
    
        self._processed_data_path = os.path.join(self._processed_root,self._file_name+".h5")
    
        self._length_log10_model_path = os.path.join(self._split_root,"canonical_stats",self._file_name,
                                                     'length_gaussian',"length_log10_gaussian_model.tsv")
        
        signal_stats_root = os.path.join(self._split_root,"canonical_stats",self._file_name,'signal_stats')            
        self._donor_signal_stats_path = os.path.join(signal_stats_root,"donor_signal_stats.tsv")
        self._acceptor_signal_stats_path = os.path.join(signal_stats_root,"acceptor_signal_stats.tsv")

    @property
    def train_val_name(self):
        return self._train_val_name
    
    @property
    def test_name(self):
        return self._test_name
        
    @property
    def split_root(self):
        return self._split_root
        
    @property
    def region_table_path(self):
        return self._region_table_path

    @property
    def fasta_path(self):
        return self._fasta_path
    
    @property
    def answer_path(self):
        return self._answer_path
    
    @property
    def processed_data_path(self):
        return self._processed_data_path
    
    @property
    def length_log10_model_path(self):
        return self._length_log10_model_path
    
    @property
    def donor_signal_stats_path(self):
        return self._donor_signal_stats_path
    
    @property
    def acceptor_signal_stats_path(self):
        return self._acceptor_signal_stats_path
            
    @property
    def main_kwargs_path(self):
        return self._main_kwargs_path
        