import os
import sys
import torch
import pandas as pd
import deepdish as dd
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence
from multiprocessing import Pool,cpu_count
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder, print_progress, write_gff, write_json
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES, BASIC_GENE_MAP
from sequence_annotation.genome_handler.region_extractor import GeneInfoExtractor
from sequence_annotation.genome_handler.sequence import PLUS
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.ann_seq_processor import vecs2seq
from sequence_annotation.preprocess.utils import read_region_table
from sequence_annotation.process.flip_and_rename_coordinate import flip_and_rename_gff
from sequence_annotation.process.utils import get_seq_mask
from sequence_annotation.process.inference import BasicInference, SeqAnnInference, ann2one_hot


def ann_vecs2gene_info(channel_order, gene_info_extractor,
                       chrom_ids, lengths,ann_vecs):
    """Convert annotation vectors to dictionay of SeqInformation of region data"""
    seq_info_container = SeqInfoContainer()
    for chrom_id, length, ann_vec in zip(chrom_ids, lengths, ann_vecs):
        one_hot_vec = ann2one_hot(ann_vec, length)
        ann_seq = vecs2seq(one_hot_vec, chrom_id, PLUS, channel_order)
        info = gene_info_extractor.extract_per_seq(ann_seq)
        seq_info_container.add(info)
    return seq_info_container


class AnnVecGffConverter:
    def __init__(self, channel_order, gene_info_extractor):
        """
        The converter fix annotation vectors by their DNA sequences' information and convert to GFF format
        Parameters:
        ----------
        channel_order : list of str
            Channel order of annotation vector
        gene_info_extractor : GeneInfoExtractor
            The GeneInfoExtractor
        """
        self.channel_order = channel_order
        self.gene_info_extractor = gene_info_extractor

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['channel_order'] = self.channel_order
        config['gene_info_extractor'] = self.gene_info_extractor.get_config()
        return config

    def convert(self, chrom_ids, lengths, ann_vecs):
        """Convert annotation vectors to GFF about region data"""
        gene_info = ann_vecs2gene_info(self.channel_order,
                                       self.gene_info_extractor,
                                       chrom_ids,lengths, ann_vecs)

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


def simple_output_to_vectors(outputs):
    """Convert vectors in dictionary to torch's tensors for each attributes"""
    data = {}
    for key in ['lengths', 'chrom_ids']:
        data[key] = outputs[key]
    outputs = outputs['outputs'].transpose(1, 2)
    outputs = pad_sequence(outputs, batch_first=True)
    data['outputs'] = outputs.transpose(2, 1).cuda()
    data['masks'] = get_seq_mask(data['lengths']).cuda()
    return data
    

def convert_output_to_gff(raw_outputs,region_table,
                          raw_plus_gff_path,gff_path,
                          inference,ann_vec_gff_converter):
    onehot_list = []
    output_vec_list = []
    for index,raw_output in enumerate(raw_outputs):
        print_progress("{}% of data have been processed".format(int(100 * index / len(raw_outputs))))
        output = simple_output_to_vectors(raw_output)
        output_list.append(output)
        onehot_vecs = inference(output['outputs'],output['masks']).cpu().numpy()
        output_vec_list.append(onehot_vecs)
    arg_list = []
    for onehot_vecs,output in zip(output_vec_list,output_list):
        arg_list.append((output['chrom_ids'],output['lengths'], onehot_vecs))
            
    with Pool(processes=cpu_count()) as pool:
        gffs = pool.starmap(ann_vec_gff_converter.convert, arg_list)
    
    gff = pd.concat(gffs).sort_values(by=['chr','start','end','strand'])
    redefined_gff = flip_and_rename_gff(gff,region_table)
    write_gff(gff, raw_plus_gff_path)
    write_gff(redefined_gff, gff_path)


def main(saved_root,raw_signal_path,region_table_path,
         use_native=True,**kwargs):
    raw_outputs = dd.io.load(raw_signal_path)
    region_table = read_region_table(region_table_path)
    config_path = os.path.join(saved_root, "ann_vec_gff_converter_config.json")
    gff_path = os.path.join(saved_root, 'converted.gff')
    converter = build_ann_vec_gff_converter()
    if use_native:
        inference_ = BasicInference([0,1,2])
    else:
        inference_ = SeqAnnInference()
    raw_plus_gff_path = '.'.join(gff_path.split('.')[:-1]) + "_raw_plus.gff3"
    config = converter.get_config()
    write_json(config, config_path)
    convert_output_to_gff(raw_outputs, region_table,
                          raw_plus_gff_path,gff_path,
                          inference_, converter,**kwargs)


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert raw output to GFF')
    parser.add_argument("--saved_root",required=True,
                        help="Root to save file")
    parser.add_argument("--raw_signal_path",required=True,
                        help="The path of raw signal file in h5 format")
    parser.add_argument("--region_table_path",required=True,
                        help="The path of region data")
    parser.add_argument("-g","--gpu_id",type=int,default=0,
                        help="GPU to used")
    parser.add_argument("--transcript_threshold", type=float, default=0.5)
    parser.add_argument("--intron_threshold", type=float, default=0.5)
    parser.add_argument("--use_native", action="store_true")
    args = parser.parse_args()
    create_folder(args.saved_root)
    setting = vars(args)
    setting_path = os.path.join(args.saved_root, "inference_setting.json")
    write_json(setting, setting_path)
    kwargs = dict(setting)
    del kwargs['saved_root']
    del kwargs['raw_signal_path']
    del kwargs['gpu_id']
    del kwargs['region_table_path']

    with torch.cuda.device(args.gpu_id):
        main(args.saved_root, args.raw_signal_path, args.region_table_path,
             **kwargs)
