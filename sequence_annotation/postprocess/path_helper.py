import os
import sys
sys.path.append(os.path.dirname(__file__)+"/..")
from ..utils.utils import read_json,get_file_name
from ..preprocess.utils import get_data_names

class PathHelper:
    def __init__(self,raw_data_root,processed_root):
        self._raw_data_root = raw_data_root
        self._processed_root = processed_root
        self.split_root=os.path.join(raw_data_root,'split/single_strand_data/split_with_strand')
        self.region_table_path = os.path.join(raw_data_root,'processed/result/region_id_conversion.tsv')
        self._test_name = get_file_name(read_json(os.path.join(self.split_root,'train_val_test_path.json'))['test_path'])
        self._data_usage = get_data_names(self.split_root)
        train_val_test_path=read_json(os.path.join(self.split_root,'train_val_test_path.json'))
        self.train_val_name = get_file_name(train_val_test_path['train_val_path'])
        self.test_name = get_file_name(train_val_test_path['test_path'])
        
        signal_stats_root = os.path.join(self.split_root,"canonical_stats",
                                         self.train_val_name,'signal_stats')
        
        self._donor_signal_stats_path = os.path.join(signal_stats_root,"donor_signal_stats.tsv")
        self._acceptor_signal_stats_path = os.path.join(signal_stats_root,"acceptor_signal_stats.tsv")

    @property
    def donor_signal_stats_path(self):
        return self._donor_signal_stats_path
    
    @property
    def acceptor_signal_stats_path(self):
        return self._acceptor_signal_stats_path
            
    def get_main_kwargs_path(self):
        path = os.path.join(self._raw_data_root,'main_kwargs.csv')
        return path
        
    def get_file_name(self,trained_id,usage=None):
        if usage is not None:
            name = self._data_usage[trained_id][usage]
        else:
            name = trained_id 
        return name
    
    def get_fasta_path(self,trained_id,usage=None):
        name = self.get_file_name(trained_id,usage)
        path = os.path.join(self.split_root,'fasta',name+".fasta")
        return path
    
    def get_answer_path(self,trained_id,usage=None,on_double_strand=False):
        name = self.get_file_name(trained_id,usage)
        if on_double_strand:
            path = os.path.join(self.split_root,'gff',name+"_canonical_double_strand.gff3")
        else:
            path = os.path.join(self.split_root,'gff',name+"_canonical.gff3")
        return path
    
    def get_processed_data_path(self,trained_id,usage=None):
        name = self.get_file_name(trained_id,usage)
        path = os.path.join(self._processed_root,name+".h5")
        return path
    
    def get_length_log10_model_path(self,trained_id,usage=None):
        name = self.get_file_name(trained_id,usage)
        path = os.path.join(self.split_root,"canonical_stats",name,
                            'length_gaussian',"length_log10_gaussian_model.tsv")
        return path
    
    
