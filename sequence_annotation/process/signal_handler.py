import os
import pandas as pd
from multiprocessing import Pool,cpu_count
from ..utils.utils import create_folder
from ..file_process.utils import BASIC_GENE_ANN_TYPES, BASIC_GENE_MAP,PLUS
from ..file_process.utils import read_gff,write_gff,BASIC_GFF_FEATURES
from ..file_process.get_region_table import read_region_table
from ..genome_handler.region_extractor import GeneInfoExtractor
from ..genome_handler.seq_container import SeqInfoContainer
from ..genome_handler.ann_seq_processor import vecs2seq
from .callback import Callback, Callbacks, get_prefix
from .performance import compare_and_save
from .flip_and_rename_coordinate import flip_and_rename_gff


class AnnVecGffConverter:
    def __init__(self, channel_order, gene_info_extractor):
        """
        The converter create GFF from annotation vectors
        Parameters:
        ----------
        channel_order : list of str
            Channel order of annotation vector
        gene_info_extractor : GeneInfoExtractor
            The GeneInfoExtractor
        """
        self._channel_order = channel_order
        self._gene_info_extractor = gene_info_extractor

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['channel_order'] = self._channel_order
        config['gene_info_extractor'] = self._gene_info_extractor.get_config()
        return config

    def convert(self, chrom_ids, lengths, ann_vecs):
        """Convert annotation vectors to GFF about region data"""
        gene_info = SeqInfoContainer()
        for chrom_id, length, ann_vec in zip(chrom_ids, lengths, ann_vecs):
            ann_seq = vecs2seq(ann_vec, chrom_id, PLUS, self._channel_order)
            info = self._gene_info_extractor.extract_per_seq(ann_seq)
            gene_info.add(info)
        gff = gene_info.to_gff()
        return gff


def build_ann_vec_gff_converter(channel_order=None, simply_map=None):
    if channel_order is None:
        channel_order = BASIC_GENE_ANN_TYPES
    if simply_map is None:
        simply_map = BASIC_GENE_MAP
    gene_info_extractor = GeneInfoExtractor(simply_map)
    ann_vec_gff_converter = AnnVecGffConverter(channel_order,
                                               gene_info_extractor)
    return ann_vec_gff_converter


def convert_signal_to_gff(results,region_table,
                          raw_plus_gff_path,gff_path,
                          ann_vec_gff_converter):
    arg_list = []
    for index,result in enumerate(results):
        onehot_vecs = result['predict'].cpu().numpy()
        arg_list.append((result['chrom'],result['length'], onehot_vecs))
            
    with Pool(processes=cpu_count()) as pool:
        gffs = pool.starmap(ann_vec_gff_converter.convert, arg_list)
        
    gff = pd.concat(gffs).sort_values(by=['chr','start','end','strand'])
    gff = gff[gff['feature'].isin(BASIC_GFF_FEATURES)]
    redefined_gff = flip_and_rename_gff(gff,region_table)
    write_gff(gff, raw_plus_gff_path)
    write_gff(redefined_gff, gff_path)

    
class _SignalGFFConverter(Callback):
    def __init__(self,region_table_path,ann_vec_gff_converter,output_root,prefix=None):
        """
        Parameters:
        ----------
        output_root: str
            The root to save GFF result and config
        region_table_path: str
            The Path about region table
        signal_saver_storage_root: SignalSaver
            The Callback to save signal
        ann_vec_gff_converter: AnnVecGffConverter
            The Converter which convert annotation vectors to GFF
        """
        self._prefix = get_prefix(prefix)
        self._output_root = output_root
        self._region_table_path = region_table_path
        self._ann_vec_gff_converter = ann_vec_gff_converter
        self._region_table = read_region_table(self._region_table_path)
        self._double_strand_gff_path = os.path.join(
            self._output_root, "{}predict_double_strand.gff3".format(self._prefix))
        self._plus_strand_gff_path = os.path.join(
            self._output_root, "{}predict_plus_strand.gff3".format(self._prefix))
        self._output_path_json = None
        self._has_finish = True
        self._index = None
        self._signals_list = []

    @property
    def gff_path(self):
        return self._double_strand_gff_path
    
    def get_config(self):
        config = super().get_config()
        config['prefix'] = self._prefix
        config['path'] = self._output_root
        config['output_root'] = self._output_root
        config['ann_vec_gff_converter'] = self._ann_vec_gff_converter.get_config()
        config['region_table_path'] = self._region_table_path
        return config

    def on_work_begin(self, worker, **kwargs):
        self._has_finish = False

    def on_epoch_begin(self, **kwargs):
        self._signals_list = []
        self._index = 0
        create_folder(self._output_root)
        
    def on_batch_end(self, seq_data, masks, predicts, metric, outputs, **kwargs):
        if not self._has_finish:
            signals = {
                'predict': predicts.get('annotation'),
                'chrom': seq_data.get('id'),
                'length': seq_data.get('length'),
            }
            self._signals_list.append(signals)
            self._index += 1

    def on_epoch_end(self, **kwargs):
        self._has_finish = True
        
    def on_work_end(self, **kwargs):
        signals = self._signals_list
        convert_signal_to_gff(signals, self._region_table,
                              self._plus_strand_gff_path,
                              self._double_strand_gff_path,
                              self._ann_vec_gff_converter)

class _GFFCompare(Callback):
    def __init__(self,region_table_path, predict_path,answer_path,output_root):
        self._output_root = output_root
        self._predict_path = predict_path
        self._answer_path = answer_path
        self._region_table_path = region_table_path
        self._region_table = read_region_table(self._region_table_path)

    def get_config(self):
        config = super().get_config()
        config['output_root'] = self._output_root
        config['predict_path'] = self._predict_path
        config['answer_path'] = self._answer_path
        config['region_table_path'] = self._region_table_path
        return config

    def on_work_end(self, **kwargs):
        predict = read_gff(self._predict_path)
        answer = read_gff(self._answer_path)
        compare_and_save(predict, answer, self._region_table,self._output_root,multiprocess=cpu_count())


class SignalHandler(Callback):
    def __init__(self,signal_gff_converter,gff_compare=None):
        self._callbacks = Callbacks([signal_gff_converter])
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
    def __init__(self,region_table_path,ann_vec_gff_converter, output_root, prefix=None,answer_gff_path=None):
        self._output_root = output_root
        self._prefix = prefix
        self._region_table_path = region_table_path
        self._ann_vec_gff_converter = ann_vec_gff_converter
        self._answer_gff_path = None
        self._answer_gff_path = answer_gff_path
        self._add_comparer =  self._answer_gff_path is not None

    def build(self):
        gff_compare = None
        converter = _SignalGFFConverter(self._region_table_path,self._ann_vec_gff_converter,
                                        self._output_root,prefix=self._prefix)
        if self._add_comparer:
            gff_compare = _GFFCompare(self._region_table_path,converter.gff_path,
                                      self._answer_gff_path,self._output_root)
        signal_handler = SignalHandler(converter, gff_compare)
        return signal_handler


def create_signal_handler(ann_types,region_table_path,root,prefix=None,answer_gff_path=None):
    converter = build_ann_vec_gff_converter(ann_types)
    builder = SignalHandlerBuilder(region_table_path,converter,root,
                                   prefix=prefix,answer_gff_path=answer_gff_path)
    signal_handler = builder.build()
    return signal_handler
