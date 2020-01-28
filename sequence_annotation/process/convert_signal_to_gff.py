import os
import sys
import torch
import pandas as pd
import numpy as np
import deepdish as dd
from argparse import ArgumentParser
torch.backends.cudnn.benchmark = True
from keras.preprocessing.sequence import pad_sequences
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../..")))
from sequence_annotation.utils.utils import create_folder, print_progress, write_gff, write_json
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES,BASIC_GENE_MAP,read_region_table
from sequence_annotation.genome_handler.region_extractor import GeneInfoExtractor
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.ann_seq_processor import get_background,vecs2seq
from sequence_annotation.preprocess.flip_coordinate import flip_gff
from sequence_annotation.preprocess.utils import RNA_TYPES
from sequence_annotation.process.utils import get_seq_mask
from sequence_annotation.process.inference import create_basic_inference,seq_ann_inference,ann_vec2one_hot_vec
from sequence_annotation.process.boundary_process import get_fixed_intron_boundary,get_exon_boundary
from sequence_annotation.process.boundary_process import fix_splice_pairs,get_splice_pairs,find_substr
basic_inference = create_basic_inference()

def _create_ann_seq(chrom,length,ann_types):
    ann_seq = AnnSequence(ann_types,length)
    ann_seq.id = ann_seq.chromosome_id = chrom
    ann_seq.strand = 'plus'
    return ann_seq

class GffPostProcessor:
    def __init__(self,gene_info_extractor,
                 donor_site_pattern=None,acceptor_site_pattern=None,
                 length_threshold=None,distance=None,
                 gene_length_threshold=None):
        """
        The post-processor fixes annotatioin result in GFF format by its DNA sequence information
        distance : int
            Valid distance  
        donor_site_pattern : str (default : GT)
            Regular expression of donor site
        acceptor_site_pattern : str (default : AG)
            Regular expression of acceptor site
        """
        self.gene_info_extractor = gene_info_extractor
        self.extractor = RegionExtractor()
        self.donor_site_pattern = donor_site_pattern or 'GT'
        self.acceptor_site_pattern = acceptor_site_pattern or 'AG'
        self.length_threshold = length_threshold or 0
        self.distance = distance or 0
        self.gene_length_threshold = gene_length_threshold or 0
        self._ann_types = ['exon','intron','other']
        
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['gene_info_extractor'] = self.gene_info_extractor.get_config()
        config['donor_site_pattern'] = self.donor_site_pattern
        config['acceptor_site_pattern'] = self.acceptor_site_pattern
        config['length_threshold'] = self.length_threshold
        config['gene_length_threshold'] = self.gene_length_threshold
        config['distance'] = self.distance
        config['ann_types'] = self._ann_types
        return config
        
    def _validate_gff(self,gff):
        strand = list(set(gff['strand']))
        if len(strand)!=1 or strand[0] != '+':
            raise Exception("Got strands, {}".format(strand))
        
    def _create_ann_seq(self,chrom,length,gff):
        ann_seq = _create_ann_seq(chrom,length,self._ann_types)
        exon_introns = gff[gff['feature'].isin(['exon','intron'])].to_dict('record')
        for item in exon_introns:
            ann_seq.add_ann(item['feature'],1,item['start']-1,item['end']-1)
        other = get_background(ann_seq,['exon','intron'])
        ann_seq.add_ann('other',other)
        return ann_seq

    def _process_ann(self,info):
        info = info.to_data_frame()
        info = info.assign(length=info['end'] - info['start'] + 1)
        fragments = info[info['length'] < self.length_threshold].sort_values(by='length')
        blocks = info[info['length'] >= self.length_threshold]
        if len(blocks)==0:
            raise Exception()
        index=0
        while True:
            if len(fragments) == 0:
                break
            fragments = fragments.reset_index(drop=True)
            fragment = dict(fragments.loc[index])
            rights = blocks[blocks['start'] == fragment['end']+1].to_dict('record')
            lefts = blocks[blocks['end'] == fragment['start']-1].to_dict('record')
            feature = None
            if len(lefts) > 0:
                left = lefts[0]
                feature = left['ann_type']
            elif len(rights) > 0:
                right = rights[0]
                feature = right['ann_type']
            if feature is None:
                index+=1
            else:
                fragment['ann_type'] = feature
                blocks = blocks.append(fragment,ignore_index=True)
                fragments=fragments.drop(index)
                index=0
        blocks_ = SeqInfoContainer()
        blocks_.from_dict({'data':blocks.to_dict('record'),'note':None})
        return blocks_

    def _process_transcript(self,rna,intron_boundarys,ann_seq):
        rna_boundary = rna['start'],rna['end']
        length = rna['end']-rna['start']+1
        if length >= self.gene_length_threshold:
            rna_intron_boundarys = []
            for intron_boundary in intron_boundarys:
                if rna['start'] <= intron_boundary[0] <= intron_boundary[1] <= rna['end']:
                    rna_intron_boundarys.append(intron_boundary)
            rna_intron_boundarys = get_fixed_intron_boundary(rna_boundary,rna_intron_boundarys)
            rna_exon_boundarys = get_exon_boundary(rna_boundary,rna_intron_boundarys)
            for boundary in rna_exon_boundarys:
                ann_seq.add_ann('exon',1,boundary[0]-1,boundary[1]-1)
            for boundary in rna_intron_boundarys:
                ann_seq.add_ann('intron',1,boundary[0]-1,boundary[1]-1)

    def process(self,chrom,length,seq,gff):
        """
        The method fixes annotatioin result in GFF format by its DNA sequence information
        Parameters:
        ----------
        chrom : str
            Chromosome id to be chosen
        seq : str
            DNA sequence which its direction is 5' to 3'
        length : int
            Length of chromosome
        gff : pd.DataFrame    
            GFF data about exon and intron
        Returns:
        ----------
        SeqInfoContainer
        """
        self._validate_gff(gff)
        gff = gff[gff['chr']==chrom]
        ann_seq = self._create_ann_seq(chrom,length,gff)
        info = self.extractor.extract(ann_seq)
        gff = self._process_ann(info).to_gff()
        ann_seq = self._create_ann_seq(chrom,length,gff)
        gff = self.gene_info_extractor.extract_per_seq(ann_seq).to_gff()
        splice_pairs = get_splice_pairs(gff)
        ann_donor_sites = [site + 1 for site in find_substr(self.donor_site_pattern,seq)]
        ann_acceptor_sites = [site + 1 for site in find_substr(self.acceptor_site_pattern,seq,False)]
        intron_boundarys = fix_splice_pairs(splice_pairs,ann_donor_sites,ann_acceptor_sites,self.distance)
        fixed_ann_seq = _create_ann_seq(chrom,length,self._ann_types)
        transcripts = gff[gff['feature'].isin(RNA_TYPES)].to_dict('record')
        for transcript in transcripts:
            self._process_transcript(transcript,intron_boundarys,fixed_ann_seq)
        other = get_background(fixed_ann_seq,['exon','intron'])
        fixed_ann_seq.add_ann('other',other)
        info = self.gene_info_extractor.extract_per_seq(fixed_ann_seq)
        gff = info.to_gff()
        return gff

class AnnVecGffConverter:
    def __init__(self,channel_order,gff_post_processor):
        """
        The converter fix annotation vectors by their DNA sequences' information and convert to GFF format
        Parameters:
        ----------
        channel_order : list of str
            Channel order of annotation vector
        gff_post_processor : GffPostProcessor
            The processor fix annotatioin result in GFF format 
        """
        self.channel_order = channel_order
        self.gff_post_processor = gff_post_processor

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['channel_order'] = self.channel_order
        config['gff_post_processor'] = self.gff_post_processor.get_config()
        return config

    def _vecs2info_dict(self,chrom_ids,lengths,ann_vecs):
        """Convert annotation vectors to dictionay of SeqInformation of region data"""
        seq_info_dict = {}
        for chrom_id,ann_vec, length in zip(chrom_ids,ann_vecs,lengths):
            one_hot_vec = ann_vec2one_hot_vec(ann_vec,length)
            ann_seq = vecs2seq(one_hot_vec,chrom_id,'plus',self.channel_order)
            info = self.gff_post_processor.gene_info_extractor.extract_per_seq(ann_seq)
            seq_info_dict[chrom_id] = info
        return seq_info_dict
    
    def vecs2raw_gff(self,chrom_ids,lengths,ann_vecs):
        """Convert annotation vectors to GFF about region data without fixing"""
        returned = SeqInfoContainer()
        info_dict = self._vecs2info_dict(chrom_ids,lengths,ann_vecs)
        for seq in info_dict.values():
            returned.add(seq)
        return returned.to_gff()

    def _info_dict2processed_gff(self,chrom_ids,lengths,dna_seqs,seq_info_dict):
        """Convert dictionay of SeqInformation to SeqInfoContainer about processed region data"""
        returned = []
        for chrom_id,dna_seq, length in zip(chrom_ids,dna_seqs,lengths):
            info = seq_info_dict[chrom_id]
            if len(info) > 0:
                gff = info.to_gff()
                try:
                    info = self.gff_post_processor.process(chrom_id,length,dna_seq,gff)
                    returned.append(info)
                except EmptyContainerException:
                    pass
        returned = pd.concat(returned)
        return returned
    
    def vecs2processed_gff(self,chrom_ids,lengths,dna_seqs,ann_vecs):
        """Convert annotation vectors to GFF about fixed region data"""
        info_dict = self._vecs2info_dict(chrom_ids,lengths,ann_vecs)
        fixed_gff = self._info_dict2processed_gff(chrom_ids,lengths,dna_seqs,info_dict)
        return fixed_gff
    
    def vecs2gff(self,chrom_ids,lengths,ann_vecs):
        """Convert annotation vectors to GFF about region data"""
        p = self.gff_post_processor
        if p.distance==0 and p.length_threshold==0 and p.gene_length_threshold == 0:
            gff = self.vecs2raw_gff(chrom_ids,lengths,ann_vecs)
        else:
            gff = self.vecs2processed_gff(chrom_ids,lengths,ann_vecs)
        return gff

def build_ann_vec_gff_converter(channel_order,simply_map,**kwargs):
    gene_info_extractor = GeneInfoExtractor(simply_map)
    gff_post_processor = GffPostProcessor(gene_info_extractor,**kwargs)
    ann_vec_gff_converter = AnnVecGffConverter(channel_order,gff_post_processor)
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
        data['outputs'] += list(np.transpose(item['outputs'],(0,2,1)))
    for key in ['dna_seqs', 'lengths','chrom_ids']:
        data[key] = np.concatenate(data[key])
    data['outputs'] = pad_sequences(data['outputs'],padding='post',dtype='float32')
    data['outputs'] = torch.Tensor(data['outputs'].transpose(0,2,1)).cuda()
    data['masks'] = get_seq_mask(data['lengths'],to_cuda=True)
    return data

def _convert_vectors_to_gff(chrom_ids,lengths,masks,dna_seqs,ann_vecs,
                          ann_vec_gff_converter,use_native=True,
                          transcript_threshold=None,intron_threshold=None):
    """Convert raw output's torch tensor to GFF dataframe"""
    if use_native:
        ann_vecs = basic_inference(ann_vecs,masks)
    else:
        ann_vecs = seq_ann_inference(ann_vecs,masks,
                                     transcript_threshold=transcript_threshold,
                                     intron_threshold=intron_threshold)
    ann_vecs = ann_vecs.cpu().numpy()
    info = ann_vec_gff_converter.vecs2gff(chrom_ids,lengths,ann_vecs)
    return info

def convert_raw_output_to_gff(raw_outputs,region_table,config_path,gff_path,
                              ann_vec_gff_converter,**kwargs):
    config = ann_vec_gff_converter.get_config()
    write_json(config,config_path)
    outputs = _convert_raw_output_to_vectors(raw_outputs)
    gff = _convert_vectors_to_gff(outputs['chrom_ids'],outputs['lengths'],outputs['masks'],
                                outputs['dna_seqs'],outputs['outputs'],
                                ann_vec_gff_converter=ann_vec_gff_converter,**kwargs)
    redefined_gff = flip_gff(gff,region_table)
    write_gff(redefined_gff,gff_path)

def convert_main(saved_root,raw_signal_path,region_path,distance=None,
                 length_threshold=None,gene_length_threshold=None,**kwargs):
    raw_outputs = dd.io.load(raw_signal_path)
    region_table = read_region_table(region_path)
    config_path = os.path.join(saved_root,"ann_vec_gff_converter_config.json")
    gff_path = os.path.join(saved_root,'predicted.gff')
    converter = build_ann_vec_gff_converter(BASIC_GENE_ANN_TYPES,BASIC_GENE_MAP,
                                            distance=distance,length_threshold=length_threshold,
                                            gene_length_threshold=gene_length_threshold)
    convert_raw_output_to_gff(raw_outputs,region_table,
                              config_path,gff_path,converter,**kwargs)

if __name__ == '__main__':    
    parser = ArgumentParser(description='Convert raw output to GFF')
    parser.add_argument("--saved_root",help="Root to save file",required=True)
    parser.add_argument("--raw_signal_path",help="The path of raw signal file in h5 format",required=True)
    parser.add_argument("--region_path",help="The path of region data",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--transcript_threshold",type=float,default=0.5)
    parser.add_argument("--intron_threshold",type=float,default=0.5)
    parser.add_argument("--length_threshold",type=int,default=0)
    parser.add_argument("--distance",type=int,default=0)
    parser.add_argument("--gene_length_threshold",type=int,default=0)
    parser.add_argument("--use_native",action="store_true")
    
    args = parser.parse_args()
    create_folder(args.saved_root)
    setting = vars(args)
    setting_path = os.path.join(args.saved_root,"inference_setting.json")
    write_json(setting,setting_path)
    kwargs = dict(setting)
    del kwargs['saved_root']
    del kwargs['raw_signal_path']
    del kwargs['gpu_id']
    del kwargs['region_path']
        
    with torch.cuda.device(args.gpu_id):
        convert_main(args.saved_root,args.raw_signal_path,args.region_path,**kwargs)
