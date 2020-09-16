import os
import torch
import pandas as pd
import deepdish as dd
from ..utils.utils import read_gff,create_folder,read_json,write_json
from ..preprocess.utils import read_region_table
from .callback import Callback, Callbacks, get_prefix
from .convert_signal_to_gff import convert_output_to_gff
from .performance import compare_and_save
from .convert_signal_to_gff import build_ann_vec_gff_converter

class _SignalSaver(Callback):
    def __init__(self, saved_root, prefix=None):
        self._prefix = get_prefix(prefix)
        self._saved_root = saved_root
        self._region_ids = None
        self._storage_root = os.path.join(self._saved_root,'signal_saver_storage')
        self._output_root = os.path.join(self._storage_root, '{}output').format(self._prefix)
        self._region_id_path = os.path.join(self._storage_root, '{}region_ids.tsv').format(self._prefix)
        self._output_list_path = os.path.join(self._storage_root, '{}output_path.json').format(self._prefix)
        self._output_path_json = None
        self._has_finish = True
        self._index = None
        
    @property
    def output_list_path(self):
        return self._output_list_path

    @property
    def region_id_path(self):
        return self._region_id_path

    def get_config(self):
        config = super().get_config()
        config['prefix'] = self._prefix
        config['path'] = self._saved_root
        return config

    def on_work_begin(self, worker, **kwargs):
        self._model = worker.model
        self._has_finish = False

    def on_epoch_begin(self, **kwargs):
        self._region_ids = []
        self._output_path_json = []
        self._index = 0
        create_folder(self._storage_root)
        create_folder(self._output_root)
        

    def on_batch_end(self, outputs, seq_data, **kwargs):
        if not self._has_finish:
            raw_output_path = os.path.join(self._output_root, '{}raw_output_{}.h5').format(self._prefix,self._index)
            self._output_path_json.append(raw_output_path)
            raw_outputs = {
                'outputs': torch.Tensor(outputs),
                'chrom_ids': seq_data.get('ids'),
                'lengths': seq_data.get('lengths')
            }
            dd.io.save(raw_output_path, raw_outputs)
            self._region_ids += seq_data.get('ids')
            self._index += 1

    def on_epoch_end(self, **kwargs):
        self._has_finish = True

    def on_work_end(self, **kwargs):
        region_ids = {'region_id': self._region_ids}
        pd.DataFrame.from_dict(region_ids).to_csv(self.region_id_path,index=None)
        write_json(self._output_path_json,self.output_list_path)

        
class _SignalGffConverter(Callback):
    def __init__(self,saved_root,region_table_path,output_list_path,
                 ann_vec_gff_converter,prefix=None):
        """
        Parameters:
        ----------
        saved_root: str
            The root to save GFF result and config
        region_table_path: str
            The Path about region table
        signal_saver_storage_root: SignalSaver
            The Callback to save signal
        ann_vec_gff_converter: AnnVecGffConverter
            The Converter which convert annotation vectors to GFF
        """
        set_prefix(self, prefix)
        self._saved_root = saved_root
        self._region_table_path = region_table_path
        self._ann_vec_gff_converter = ann_vec_gff_converter
        self._region_table = read_region_table(self._region_table_path)
        self._output_list_path = output_list_path
        self._double_strand_gff_path = os.path.join(
            self._saved_root, "{}predict_double_strand.gff3".format(self._prefix))
        self._plus_strand_gff_path = os.path.join(
            self._saved_root, "{}predict_plus_strand.gff3".format(self._prefix))

    @property
    def gff_path(self):
        return self._double_strand_gff_path

    def get_config(self):
        config = super().get_config()
        config['saved_root'] = self._saved_root
        config['ann_vec_gff_converter'] = self._ann_vec_gff_converter.get_config()
        config['region_table_path'] = self._region_table_path
        return config

    def on_work_begin(self, worker):
        self._inference = worker.executor.inference
    
    def on_work_end(self, **kwargs):
        raw_predicts = []
        for path in read_json(self._output_list_path):
            data = dd.io.load(path)
            raw_predicts.append(data)
        convert_output_to_gff(raw_predicts, self._region_table,
                              self._plus_strand_gff_path,
                              self._double_strand_gff_path,
                              self._inference,
                              self._ann_vec_gff_converter)


class _GFFCompare(Callback):
    def __init__(self, saved_root, region_table_path,
                 predict_path,answer_path,region_id_path):
        self._saved_root = saved_root
        self._predict_path = predict_path
        self._answer_path = answer_path
        self._region_table_path = region_table_path
        self._region_id_path = region_id_path
        self._region_table = read_region_table(self._region_table_path)

    def get_config(self):
        config = super().get_config()
        config['saved_root'] = self._saved_root
        config['predict_path'] = self._predict_path
        config['answer_path'] = self._answer_path
        config['region_table_path'] = self._region_table_path
        config['region_id_path'] = self._region_id_path
        return config

    def on_work_end(self, **kwargs):
        predict = read_gff(self._predict_path)
        answer = read_gff(self._answer_path)
        region_ids = set(pd.read_csv(self._region_id_path)['region_id'])
        part_region_table = self._region_table[self._region_table['ordinal_id_with_strand'].isin(region_ids)]
        compare_and_save(predict, answer, part_region_table,self._saved_root)


class SignalHandler(Callback):
    def __init__(self,signal_saver,signal_gff_converter=None,gff_compare=None):
        if gff_compare is not None and signal_gff_converter is None:
            raise Exception("To use gff_compare, the signal_gff_converter should be provided")
        self._callbacks = Callbacks([signal_saver])
        if signal_gff_converter is not None:
            self._callbacks.add(signal_gff_converter)
        if gff_compare is not None:
            self._callbacks.add(gff_compare)

    def get_config(self):
        config = super().get_config()
        config.update(self._callbacks.get_config())
        return config

    def on_work_begin(self, **kwargs):
        self._callbacks.on_work_begin(**kwargs)

    def on_work_end(self, **kwargs):
        self._callbacks.on_work_end(**kwargs)

    def on_epoch_begin(self, **kwargs):
        self._callbacks.on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs):
        self._callbacks.on_epoch_end(**kwargs)

    def on_batch_begin(self, **kwargs):
        self._callbacks.on_batch_begin(**kwargs)

    def on_batch_end(self, **kwargs):
        self._callbacks.on_batch_end(**kwargs)


class SignalHandlerBuilder:
    def __init__(self, saved_root, prefix=None):
        self._saved_root = saved_root
        self._prefix = prefix
        self._region_table_path = None
        self._ann_vec_gff_converter = None
        self._answer_gff_path = None
        self._add_converter = False
        self._add_comparer = False

    def add_converter_args(self,region_table_path,ann_vec_gff_converter):
        self._region_table_path = region_table_path
        self._ann_vec_gff_converter = ann_vec_gff_converter
        self._add_converter = True

    def add_answer_path(self, answer_gff_path):
        self.answer_gff_path = answer_gff_path
        self._add_comparer = True

    def build(self):
        signal_saver = _SignalSaver(self._saved_root, prefix=self._prefix)
        converter = None
        gff_compare = None
        if self._add_converter:
            converter = _SignalGffConverter(self._saved_root,
                                            self._region_table_path,
                                            signal_saver.output_list_path,
                                            self._ann_vec_gff_converter,
                                            prefix=self._prefix)
            if self._add_comparer:
                gff_compare = _GFFCompare(self._saved_root,
                                          self._region_table_path,
                                          converter.gff_path,
                                          self._answer_gff_path,
                                          signal_saver.region_id_path)
        signal_handler = SignalHandler(signal_saver, converter, gff_compare)
        return signal_handler
    
def get_signal_handler(ann_types,root,prefix=None,region_table_path=None,answer_gff_path=None):
    builder = SignalHandlerBuilder(root, prefix=prefix)
    if region_table_path is not None:
        converter = build_ann_vec_gff_converter(ann_types)
        builder.add_converter_args(region_table_path,converter)
    if answer_gff_path is not None:
        builder.add_answer_path(answer_gff_path)
    signal_handler = builder.build()
    return signal_handler
