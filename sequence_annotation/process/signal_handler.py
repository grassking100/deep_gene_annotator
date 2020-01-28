import os
import pandas as pd
import deepdish as dd
from ..utils.utils import gffcompare_command,read_region_table,read_gff
from .callback import Callback,Callbacks,set_prefix
from .convert_signal_to_gff import convert_raw_output_to_gff
from .performance import compare_and_save

class _SignalSaver(Callback):
    def __init__(self,saved_root,prefix=None):
        set_prefix(self,prefix)
        if saved_root is None:
            raise Exception("The saved_root should not be None")
        self._saved_root = saved_root
        self._raw_outputs = None
        self._raw_answers = None
        self._region_ids = None
        self._raw_answer_path = os.path.join(self._saved_root,'{}raw_answer.h5').format(self._prefix)
        self._raw_output_path = os.path.join(self._saved_root,'{}raw_output.h5').format(self._prefix)
        self._region_id_path = os.path.join(self._saved_root,'{}region_ids.tsv').format(self._prefix)
        self._has_finish = True
       
    @property
    def raw_answer_path(self):
        return self._raw_answer_path

    @property
    def raw_output_path(self):
        return self._raw_output_path
    
    @property
    def region_id_path(self):
        return self._region_id_path
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self._saved_root
        return config
        
    def on_work_begin(self,worker,**kwargs):
        self._model = worker.model
        self._has_finish = False

    def on_epoch_begin(self,**kwargs):
        self._raw_outputs = []
        self._raw_answers = []
        self._region_ids = []

    def on_batch_end(self,outputs,seq_data,**kwargs):
        if not self._has_finish:
            chrom_ids,dna_seqs,answers,lengths = seq_data.ids,seq_data.seqs,seq_data.answers,seq_data.lengths
            answers = answers.cpu().numpy()
            self._raw_outputs.append({'outputs':outputs,
                                      'chrom_ids':chrom_ids,
                                      'dna_seqs':dna_seqs,
                                      'lengths':lengths})

            self._raw_answers.append({'outputs':answers,
                                      'chrom_ids':chrom_ids,
                                      'dna_seqs':dna_seqs,
                                      'lengths':lengths})
            self._region_ids += chrom_ids.tolist()

    def on_epoch_end(self,**kwargs):
        self._has_finish = True
            
    def on_work_end(self,**kwargs):
        dd.io.save(self.raw_output_path, self._raw_outputs)
        dd.io.save(self.raw_answer_path, self._raw_answers)
        region_ids = {'region_id':self._region_ids}
        pd.DataFrame.from_dict(region_ids).to_csv(self.region_id_path,index=None)

class _SignalGffConverter(Callback):
    def __init__(self,saved_root,region_table_path,raw_output_path,
                 ann_vec_gff_converter,prefix=None,**kwargs):
        """
        Parameters:
        ----------
        saved_root: str
            The root to save GFF result and config
        region_table_path: str
            The Path about region table
        raw_output_path: SignalSaver
            The Callback to save signal
        ann_vec_gff_converter: AnnVecGffConverter
            The Converter which convert annotation vectors to GFF
        kwargs: dict
            The optional settings to seq_ann_inference
        """
        set_prefix(self,prefix)
        variables = [saved_root,region_table_path,raw_output_path,ann_vec_gff_converter]
        names = ['saved_root','region_table_path','raw_output_path','ann_vec_gff_converter']        
        for name,variable in zip(names,variables):
            if variable is None:
                raise Exception("The {} should not be None".format(name))
    
        self._saved_root = saved_root
        self._region_table_path = region_table_path
        self._raw_output_path = raw_output_path
        self._ann_vec_gff_converter = ann_vec_gff_converter
        self._kwargs = kwargs
        self._config_path = os.path.join(self._saved_root,"converter_config.json")
        self._region_table = read_region_table(self._region_table_path)
        
        self._predict_gff_path = os.path.join(self._saved_root,"{}gffcompare.gff3".format(self._prefix))
        
    @property
    def predict_gff_path(self):
        return self._predict_gff_path
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['saved_root'] = self._saved_root
        config['ann_vec_gff_converter'] = self._ann_vec_gff_converter.get_config()
        config['region_table_path'] = self._region_table_path
        return config
        
    def on_work_end(self,**kwargs):
        raw_predicts = dd.io.load(self._raw_output_path)
        convert_raw_output_to_gff(raw_predicts,self._region_table,
                                  self._config_path,self.predict_gff_path,
                                  self._ann_vec_gff_converter,**self._kwargs)

class _GFFCompare(Callback):
    def __init__(self,saved_root,region_table_path,predict_path,answer_path,region_id_path):
        variables = [saved_root,predict_path,answer_path,region_table_path,region_id_path]
        names = ['saved_root','predict_path','answer_path','region_table_path','region_id_path']
        for name,variable in zip(names,variables):
            if variable is None:
                raise Exception("The {} should not be None".format(name))

        self._saved_root = saved_root
        self._predict_path = predict_path
        self._answer_path = answer_path
        self._region_table_path = region_table_path
        self._region_id_path = region_id_path
        self._region_table = read_region_table(self._region_table_path)
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['saved_root'] = self._saved_root
        config['predict_path'] = self._predict_path
        config['answer_path'] = self._answer_path
        config['region_table_path'] = self._region_table_path
        config['region_id_path'] = self._region_id_path
        return config
        
    def on_work_end(self,**kwargs):
        predict = read_gff(self._predict_path)
        answer = read_gff(self._answer_path)
        chrom_lengths = {}
        region_ids = set(pd.read_csv(self._region_id_path)['region_id'])
        part_region_table = self._region_table[self._region_table['old_id'].isin(region_ids)]
        compare_and_save(predict,answer,part_region_table,self._saved_root)

class SignalHandler(Callback):
    def __init__(self,signal_saver,signal_gff_converter,gff_compare):
        self._callbacks = Callbacks([signal_saver,signal_gff_converter,gff_compare])
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config.update(self._callbacks.get_config())
        return config
        
    def on_work_begin(self,**kwargs):
        self._callbacks.on_work_begin(**kwargs)

    def on_work_end(self,**kwargs):
        self._callbacks.on_work_end(**kwargs)

    def on_epoch_begin(self,**kwargs):
        self._callbacks.on_epoch_begin(**kwargs)

    def on_epoch_end(self,**kwargs):
        self._callbacks.on_epoch_end(**kwargs)

    def on_batch_begin(self,**kwargs):
        self._callbacks.on_batch_begin(**kwargs)

    def on_batch_end(self,**kwargs):
        self._callbacks.on_batch_end(**kwargs)
        
def build_signal_handler(saved_root,region_table_path,
                         answer_gff_path,ann_vec_gff_converter,prefix=None):
    signal_saver = _SignalSaver(saved_root,prefix=prefix)
    converter = _SignalGffConverter(saved_root,region_table_path,signal_saver.raw_output_path,
                                    ann_vec_gff_converter,prefix=prefix)
    gff_compare = _GFFCompare(saved_root,region_table_path,
                              converter.predict_gff_path,
                              answer_gff_path,
                              signal_saver.region_id_path)

    signal_handler = SignalHandler(signal_saver,converter,gff_compare)
    return signal_handler
