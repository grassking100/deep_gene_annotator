import os
import deepdish as dd
from ..utils.utils import gffcompare_command,read_region_table
from .callback import Callback,Callbacks,set_prefix
from .convert_singal_to_gff import convert_raw_output_to_gff

class SignalSaver(Callback):
    def __init__(self,path,prefix=None):
        set_prefix(self,prefix)
        if path is None:
            raise Exception("The path should not be None")
        self._path = path
        self._raw_outputs = None
        self._raw_answers = None
        self._counter = None
        self._raw_answer_path = os.path.join(self._path,'{}raw_answer.h5').format(self._prefix)
       
    @property
    def raw_answer_path(self):
        return self._raw_answer_path

    @property
    def raw_output_path(self):
        path = os.path.join(self._path,'{}raw_output_{}.h5').format(self._prefix,self._counter)
        return path
        
    def on_work_begin(self,worker,**kwargs):
        self._counter = 0
        self._model = worker.model

    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self._path
        return config

    def on_epoch_begin(self,counter,**kwargs):
        self._raw_outputs = []
        self._raw_answers = []
        self._counter = counter

    def on_batch_end(self,outputs,seq_data,**kwargs):
        chrom_ids,dna_seqs,answers,lengths = seq_data.ids,seq_data.seqs,seq_data.answers,seq_data.lengths
        answers = answers.cpu().numpy()
        self._raw_outputs.append({'outputs':outputs,
                                  'chrom_ids':chrom_ids,
                                  'dna_seqs':dna_seqs,
                                  'lengths':lengths})

        if self._counter == 1:
            self._raw_answers.append({'outputs':answers,
                                      'chrom_ids':chrom_ids,
                                      'dna_seqs':dna_seqs,
                                      'lengths':lengths})

    def on_epoch_end(self,**kwargs):
        dd.io.save(self.raw_output_path , self._raw_outputs)

        if self._counter == 1:
            dd.io.save(self.raw_answer_path, self._raw_answers)

class GFFCompare(Callback):
    def __init__(self,path,region_table_path,answer_gff_path,signal_saver,
                 ann_vec2info_converter,prefix=None):
        set_prefix(self,prefix)
        if path is None:
            raise Exception("The path should not be None")
            
        if region_table_path is None:
            raise Exception("The region_table_path should not be None")
            
        if answer_gff_path is None:
            raise Exception("The answer_gff_path should not be None")
            
        if signal_saver is None:
            raise Exception("The signal_saver should not be None")
            
        if ann_vec2info_converter is None:
            raise Exception("The ann_vec2info_converter should not be None")
            
        self._counter = None
        self._path = path
        self._region_table_path = region_table_path
        self._region_table = read_region_table(self._region_table_path)
        self._signal_saver = signal_saver
        self._answer_gff_path = answer_gff_path
        self._config_path = os.path.join(self._path,"converter_config.json")
        self._ann_vec2info_converter = ann_vec2info_converter

    @property
    def signal_saver(self):
        return self._signal_saver
        
    @property
    def answer_gff_path(self):
        return self._answer_gff_path
        
    @property
    def predict_gff_path(self):
        prefix = "{}gffcompare_{}".format(self._prefix,self._counter)
        path = os.path.join(self._path,"{}.gff3".format(prefix))
        return path
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config['path'] = self._path
        config['answer_gff_path'] = self.answer_gff_path
        config['ann_vec2info_converter'] = self._ann_vec2info_converter.get_config()
        config['region_table_path'] = self._region_table_path
        return config

    def on_epoch_begin(self,counter,**kwargs):
        self._counter = counter
        
    def on_epoch_end(self,**kwargs):
        raw_predicts = dd.io.load(self._signal_saver.raw_output_path)
        convert_raw_output_to_gff(raw_predicts,self._region_table,
                                  self._config_path,self.predict_gff_path,
                                  self._ann_vec2info_converter)
        
        prefix = "{}gffcompare_{}".format(self._prefix,self._counter)
        prefix_path = os.path.join(self._path,prefix)
        gffcompare_command(self.answer_gff_path,self.predict_gff_path,prefix_path,verbose=True)
        
class SingalHandler(Callback):
    def __init__(self,singal_saver,gffcompare):
        if gffcompare.signal_saver != singal_saver:
            raise Exception()
        self.callbacks = Callbacks([singal_saver,gffcompare])
        
    def get_config(self,**kwargs):
        config = super().get_config(**kwargs)
        config.update(self.callbacks.get_config())
        return config
        
    def on_work_begin(self,**kwargs):
        self.callbacks.on_work_begin(**kwargs)

    def on_work_end(self,**kwargs):
        self.callbacks.on_work_end(**kwargs)

    def on_epoch_begin(self,**kwargs):
        self.callbacks.on_epoch_begin(**kwargs)

    def on_epoch_end(self,**kwargs):
        self.callbacks.on_epoch_end(**kwargs)

    def on_batch_begin(self,**kwargs):
        self.callbacks.on_batch_begin(**kwargs)

    def on_batch_end(self,**kwargs):
        self.callbacks.on_batch_end(**kwargs)
        
def build_singal_handler(path,region_table_path,answer_gff_path,ann_vec2info_converter,prefix=None):
    signal_saver = SignalSaver(path,prefix=prefix)
    gffcompare = GFFCompare(path,region_table_path,answer_gff_path,
                            signal_saver,ann_vec2info_converter,prefix=prefix)
    singal_handler = SingalHandler(signal_saver,gffcompare)
    return singal_handler
