import os
import sys
import torch
import numpy as np
import deepdish as dd
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../..")))
from sequence_annotation.utils.utils import create_folder, print_progress, write_gff, write_json
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES,BASIC_GENE_MAP
from sequence_annotation.genome_handler.region_extractor import GeneInfoExtractor
from sequence_annotation.genome_handler.sequence import PLUS
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.ann_seq_processor import vecs2seq
from sequence_annotation.preprocess.utils import read_region_table
from sequence_annotation.preprocess.flip_and_rename_coordinate import flip_and_rename_gff
from sequence_annotation.process.utils import get_seq_mask
from sequence_annotation.process.inference import create_basic_inference,seq_ann_inference,ann_vec2one_hot_vec

def ann_vecs2gene_info(channel_order,gene_info_extractor,chrom_ids,lengths,ann_vecs):
    """Convert annotation vectors to dictionay of SeqInformation of region data"""
    gene_info = {}
    for chrom_id, length, ann_vec in zip(chrom_ids,lengths,ann_vecs):
        one_hot_vec = ann_vec2one_hot_vec(ann_vec,length)
        ann_seq = vecs2seq(one_hot_vec,chrom_id,PLUS,channel_order)
        info = gene_info_extractor.extract_per_seq(ann_seq)
        gene_info[chrom_id] = info
    return gene_info

class AnnVecGffConverter:
    def __init__(self,channel_order,gene_info_extractor):
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

    def convert(self,chrom_ids,lengths,dna_seqs,ann_vecs):
        """Convert annotation vectors to GFF about region data"""
        gene_info = ann_vecs2gene_info(self.channel_order,self.gene_info_extractor,
                                       chrom_ids,lengths,ann_vecs)
        seq_info_container = SeqInfoContainer()
        for seq in gene_info.values():
            seq_info_container.add(seq)
        gff = seq_info_container.to_gff()
        return gff

def build_ann_vec_gff_converter(channel_order=None,simply_map=None):
    if channel_order is None:
        channel_order = BASIC_GENE_ANN_TYPES
    if simply_map is None:
        simply_map = BASIC_GENE_MAP
    gene_info_extractor = GeneInfoExtractor(simply_map)
    ann_vec_gff_converter = AnnVecGffConverter(channel_order,gene_info_extractor)
    return ann_vec_gff_converter

def _convert_raw_output_to_vectors(outputs):
    """Convert vectors in dictionary to torch's tensors for each attributes"""
    data = {}
    columns = ['dna_seqs', 'lengths', 'outputs','chrom_ids']
    for key in columns:
        data[key] = []
    for index,item in enumerate(outputs):
        print_progress("{}% of data have been processed".format(int(100*index/len(outputs))))
        for key in ['dna_seqs', 'lengths','chrom_ids']:
            data[key] += [item[key]]
        item = torch.FloatTensor(item['outputs']).transpose(1,2)
        data['outputs'] += item
    for key in ['dna_seqs', 'lengths','chrom_ids']:
        data[key] = np.concatenate(data[key])
    data['outputs'] = pad_sequence(data['outputs'],batch_first=True)
    data['outputs'] = data['outputs'].transpose(2,1).cuda()
    data['masks'] = get_seq_mask(data['lengths'],to_cuda=True)
    return data

def _convert_vectors_to_gff(chrom_ids,lengths,masks,dna_seqs,ann_vecs,
                            inference,ann_vec_gff_converter,
                            transcript_threshold=None,intron_threshold=None):
    """Convert raw output's torch tensor to GFF dataframe"""
    ann_vecs = inference(ann_vecs,masks,
                         transcript_threshold=transcript_threshold,
                         intron_threshold=intron_threshold)
    ann_vecs = ann_vecs.cpu().numpy()
    gff = ann_vec_gff_converter.convert(chrom_ids,lengths,dna_seqs,ann_vecs)
    return gff

def convert_raw_output_to_gff(raw_outputs,region_table,config_path,gff_path,
                              inference,ann_vec_gff_converter,
                              chrom_source=None,chrom_target=None,**kwargs):
    config = ann_vec_gff_converter.get_config()
    raw_plus_gff_path = '.'.join(gff_path.split('.')[:-1])+"_raw_plus.gff3"
    write_json(config,config_path)
    outputs = _convert_raw_output_to_vectors(raw_outputs)
    gff = _convert_vectors_to_gff(outputs['chrom_ids'],outputs['lengths'],outputs['masks'],
                                  outputs['dna_seqs'],outputs['outputs'],inference,
                                  ann_vec_gff_converter=ann_vec_gff_converter,**kwargs)
    redefined_gff = flip_and_rename_gff(gff,region_table,
                                        chrom_source=chrom_source,
                                        chrom_target=chrom_target)
    write_gff(gff,raw_plus_gff_path)
    write_gff(redefined_gff,gff_path)

def main(saved_root,raw_signal_path,region_table_path,use_native=True,**kwargs):
    raw_outputs = dd.io.load(raw_signal_path)
    region_table = read_region_table(region_table_path)
    config_path = os.path.join(saved_root,"ann_vec_gff_converter_config.json")
    gff_path = os.path.join(saved_root,'converted.gff')
    converter = build_ann_vec_gff_converter()
    if use_native:
        inference_ = create_basic_inference()
    else:
        inference_ = seq_ann_inference
    convert_raw_output_to_gff(raw_outputs,region_table,config_path,gff_path,
                              inference_,converter,**kwargs)

if __name__ == '__main__':    
    parser = ArgumentParser(description='Convert raw output to GFF')
    parser.add_argument("--saved_root",help="Root to save file",required=True)
    parser.add_argument("--raw_signal_path",help="The path of raw signal file in h5 format",required=True)
    parser.add_argument("--region_table_path",help="The path of region data",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--transcript_threshold",type=float,default=0.5)
    parser.add_argument("--intron_threshold",type=float,default=0.5)
    parser.add_argument("--use_native",action="store_true")
    parser.add_argument("--chrom_source",help="Valid options are old_id and new_id")
    parser.add_argument("--chrom_target",help="Valid options are old_id and new_id")
    args = parser.parse_args()
    create_folder(args.saved_root)
    setting = vars(args)
    setting_path = os.path.join(args.saved_root,"inference_setting.json")
    write_json(setting,setting_path)
    kwargs = dict(setting)
    del kwargs['saved_root']
    del kwargs['raw_signal_path']
    del kwargs['gpu_id']
    del kwargs['region_table_path']
        
    with torch.cuda.device(args.gpu_id):
        main(args.saved_root,args.raw_signal_path,args.region_table_path,**kwargs)
