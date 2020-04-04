import os
import sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/..")
from sequence_annotation.utils.utils import create_folder, write_gff, write_json, read_json, read_gff, read_fasta
from sequence_annotation.utils.utils import BASIC_GENE_MAP, read_region_table,get_gff_with_attribute
from sequence_annotation.genome_handler.region_extractor import GeneInfoExtractor
from sequence_annotation.genome_handler.sequence import AnnSequence,PLUS
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.ann_seq_processor import get_background
from sequence_annotation.preprocess.utils import RNA_TYPES,get_gff_with_intron,get_gff_with_intergenic_region
from sequence_annotation.preprocess.flip_and_rename_coordinate import flip_and_rename_gff
from sequence_annotation.postprocess.boundary_process import get_splice_pair_statuses,get_exon_boundary
from sequence_annotation.postprocess.boundary_process import get_valid_intron_boundary,get_splice_pairs,find_substr

def _create_ann_seq(chrom,length,ann_types):
    ann_seq = AnnSequence(ann_types,length)
    ann_seq.id = ann_seq.chromosome_id = chrom
    ann_seq.strand = PLUS
    return ann_seq

class GFFReviser:
    def __init__(self,gene_info_extractor,donor_pattern=None,acceptor_pattern=None,
                 length_thresholds=None,donor_distance=None,acceptor_distance=None,
                 donor_index_shift=None,acceptor_index_shift=None):
        """
        The reviser fixes annotatioin result by its DNA sequence information and length
        Parameters:
        donor_pattern : str (default : GT)
            Regular expression of donor site
        acceptor_pattern : str (default : AG)
            Regular expression of acceptor site
        length_thresholds : dict
            the length thresholds for 'exon','intron','other', and 'transcript' 
        donor_distance : int
            Valid donor site distance
        acceptor_distance : int
            Valid acceptor site distance
        donor_index_shift : int (default : 0)
            Shift value of donor site index
        acceptor_index_shift : int (default : 1)
            Shift value of acceptor site index
        """
        self.gene_info_extractor = gene_info_extractor
        self.extractor = RegionExtractor()
        self.donor_pattern = donor_pattern or 'GT'
        self.acceptor_pattern = acceptor_pattern or 'AG'
        if length_thresholds is not None:
            if set(length_thresholds.keys()) != set(['exon','intron','other','transcript']):
                raise Exception("Wrong key in length_thresholds")
        self.length_thresholds = length_thresholds
        self.donor_distance = donor_distance or 0
        self.acceptor_distance = acceptor_distance or 0
        self.donor_index_shift = donor_index_shift or 0
        self.acceptor_index_shift = acceptor_index_shift or 1
        self._ann_types = ['exon','intron','other']
        
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['gene_info_extractor'] = self.gene_info_extractor.get_config()
        config['donor_pattern'] = self.donor_pattern
        config['acceptor_pattern'] = self.acceptor_pattern
        config['length_thresholds'] = self.length_thresholds
        config['donor_distance'] = self.donor_distance
        config['acceptor_distance'] = self.acceptor_distance
        config['ann_types'] = self._ann_types
        config['donor_index_shift'] = self.donor_index_shift
        config['acceptor_index_shift'] = self.acceptor_index_shift
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

    def _filter_ann(self,seq_info_container):
        if self.length_thresholds is None:
            return seq_info_container
        dict_ = seq_info_container.to_dict()
        seq_info_list = dict_['data']
        blocks = []
        starts = []
        for seq_info in seq_info_list:
            starts.append(seq_info['start'])

        indice = np.argsort(starts)
        for index in indice:
            seq_info = seq_info_list[index]
            #If it is fragment, handle it
            length = seq_info['end'] - seq_info['start']
            if length < self.length_thresholds[seq_info['ann_type']]:
                if index==0:
                    seq_info['ann_type'] = 'other'
                    blocks.append(seq_info)
                else:
                    blocks[-1]['end'] = seq_info['end']
            else:
                blocks.append(seq_info)
                
        blocks_ = SeqInfoContainer()
        blocks_.from_dict({'data':blocks,'note':dict_['note']})
        return blocks_

    def _process_transcript(self,rna,splice_pair_statuses,ann_seq):
        rna_boundary = rna['start'],rna['end']
        intron_boundarys = get_valid_intron_boundary(rna_boundary,splice_pair_statuses)
        rna_exon_boundarys = get_exon_boundary(rna_boundary,intron_boundarys)
        for boundary in rna_exon_boundarys:
            ann_seq.add_ann('exon',1,boundary[0]-1,boundary[1]-1)
        for boundary in intron_boundarys:
            ann_seq.add_ann('intron',1,boundary[0]-1,boundary[1]-1)

    def _preprocess_gff(self,chrom_id,length,gff):
        self._validate_gff(gff)
        gff = gff[gff['chr']==chrom_id]
        gff = get_gff_with_attribute(gff)
        if 'intron' not in set(gff['feature']):
            gff = get_gff_with_intron(gff)
        if 'intergenic region' not in set(gff['feature']):
            gff = get_gff_with_intergenic_region(gff,{chrom_id:length},strands=['+'])
        gff.loc[gff['feature']=='intergenic region','feature'] = 'other'
        return gff
                
    def filter_by_threshold(self,chrom_id,length,gff):
        """
        The method removes annotation block size which is smaller then threshold
        Parameters:
        ----------
        chrom_id : str
            Chromosome id to be chosen
        length : int
            Length of chromosome
        gff : pd.DataFrame    
            GFF data in plus strand
        Returns:
        ----------
        pd.DataFrame
        """
        #Filter exon,intron,other
        gff = self._preprocess_gff(chrom_id,length,gff)
        ann_seq = self._create_ann_seq(chrom_id,length,gff)
        blocks = self.extractor.extract(ann_seq)
        gff = self._filter_ann(blocks).to_gff()
        ann_seq = self._create_ann_seq(chrom_id,length,gff)
        filtered = self.gene_info_extractor.extract_per_seq(ann_seq)
        filtered_gff = None
        if len(filtered)>0:
            #filtered.to_data_frame().to_csv('csv')
            filtered_gff = filtered.to_gff()
            #print(len(filtered_gff),len(filtered_gff.drop_duplicates()))
            #filtered_gff = filtered_gff.drop_duplicates()
            #raise Exception()
            #Filter transcript
            invalid_ids = []
            filtered_gff = get_gff_with_attribute(filtered_gff)
            #print(len(filtered_gff),len(filtered_gff.drop_duplicates()))
            transcripts = filtered_gff[filtered_gff['feature'].isin(RNA_TYPES)].to_dict('record')
            for transcript in transcripts:
                length = transcript['end']-transcript['start']+1
                if self.length_thresholds is not None:
                    if length < self.length_thresholds['transcript']:
                        #print(length,self.length_thresholds['transcript'])
                        #print("Remove "+str(transcript))
                        invalid_ids.append(transcript['parent'])
                        invalid_ids.append(transcript['id'])

            filtered_gff = filtered_gff[(~filtered_gff['parent'].isin(invalid_ids)) & (~filtered_gff['id'].isin(invalid_ids))]
        return filtered_gff

    def fix_splicing_site(self,chrom_id,length,seq,gff):
        """
        The method fixes splicing site by its sequence
        Parameters:
        ----------
        chrom_id : str
            Chromosome id to be chosen
        seq : str
            DNA sequence which its direction is 5' to 3'
        length : int
            Length of chromosome
        gff : pd.DataFrame    
            GFF data in plus strand
        Returns:
        ----------
        pd.DataFrame
        """
        gff = self._preprocess_gff(chrom_id,length,gff)
        splice_pairs = get_splice_pairs(gff)
        ann_donor_sites = find_substr(self.donor_pattern,seq,self.donor_index_shift+1)
        ann_acceptor_sites = find_substr(self.acceptor_pattern,seq,self.acceptor_index_shift+1)
        splice_pair_statuses = get_splice_pair_statuses(splice_pairs,ann_donor_sites,ann_acceptor_sites,
                                                        self.donor_distance,self.acceptor_distance)
        fixed_ann_seq = _create_ann_seq(chrom_id,length,self._ann_types)
        transcripts = gff[gff['feature'].isin(RNA_TYPES)].to_dict('record')
        for transcript in transcripts:
            self._process_transcript(transcript,splice_pair_statuses,fixed_ann_seq)
        other = get_background(fixed_ann_seq,['exon','intron'])
        fixed_ann_seq.add_ann('other',other)
        info = self.gene_info_extractor.extract_per_seq(fixed_ann_seq)
        fixed_gff = None
        if len(info)>0:
            fixed_gff = info.to_gff()
        return fixed_gff
                
    def process(self,chrom_id,length,seq,gff,methods=None):
        """
        The method fixes annotatioin result in GFF format by its DNA sequence information and threshold
        Parameters:
        ----------
        chrom_id : str
            Chromosome id to be chosen
        seq : str
            DNA sequence which its direction is 5' to 3'
        length : int
            Length of chromosome
        gff : pd.DataFrame    
            GFF data in plus strand
        method : list
            Methods: length_threshold, distance_threshold
        Returns:
        ----------
        pd.DataFrame
        """
        methods = methods or []
        for method in methods:
            if gff is None or len(gff)==0:
                break
            if method == 'length_threshold':
                gff = self.filter_by_threshold(chrom_id,length,gff)
            elif method == 'distance_threshold':
                gff = self.fix_splicing_site(chrom_id,length,seq,gff)
            else:
                raise Exception("Invalid method {}".format(method))

        return gff

def build_gff_reviser(simply_map,**kwargs):
    gene_info_extractor = GeneInfoExtractor(simply_map)
    gff_reviser = GFFReviser(gene_info_extractor,**kwargs)
    return gff_reviser
    
def revise(input_gff,region_table,fasta,reviser,methods=None):
    gff_list = []
    for chrom_id in set(input_gff['chr']):
        length = list(region_table[region_table['old_id']==chrom_id]['length'])[0]
        seq = fasta[chrom_id]
        gff = reviser.process(chrom_id,length,seq,input_gff,methods=methods)
        if gff is not None:
            gff_list.append(gff)
    gff = pd.concat(gff_list,sort=True).reset_index(drop=True)
    return gff
    
def main(saved_root,input_raw_plus_gff_path,region_table_path,fasta_path,
         length_thresholds=None,length_threshold_path=None,methods=None,**kwargs):
    create_folder(saved_root)
    raw_plus_gff_path = os.path.join(saved_root,"revised_raw_plus.gff3")
    revised_gff_path = os.path.join(saved_root,"revised.gff3")
    if length_thresholds is None and length_threshold_path is not None:
        length_thresholds = read_json(length_threshold_path)
    region_table = read_region_table(region_table_path)
    input_gff = read_gff(input_raw_plus_gff_path)
    fasta = read_fasta(fasta_path)
    reviser = build_gff_reviser(BASIC_GENE_MAP,length_thresholds=length_thresholds,**kwargs)
    config = reviser.get_config()
    config_path = os.path.join(saved_root,"reviser_config.json")
    write_json(config,config_path)
    gff = revise(input_gff,region_table,fasta,reviser,methods)
    write_gff(gff,raw_plus_gff_path)
    flipped_gff = flip_and_rename_gff(gff,region_table,chrom_source='old_id',chrom_target='old_id')
    write_gff(flipped_gff,revised_gff_path)
    
if __name__ == '__main__':    
    parser = ArgumentParser(description='Revise the GFF')
    parser.add_argument("-r","--saved_root",help="The root to save result",required=True)
    parser.add_argument("-i","--input_raw_plus_gff_path",help="The path of origin "
                        "of single-strand plus-only data in GFf format",required=True)
    parser.add_argument("-t","--region_table_path",help="The path of region data",required=True)
    parser.add_argument("-f","--fasta_path",help="The path of fasta",required=True)
    parser.add_argument("--length_threshold_path",type=str,help="The path of "
                        "length threshold for each type, written in JSON format")
    parser.add_argument("--donor_distance",type=int,default=0)
    parser.add_argument("--acceptor_distance",type=int,default=0)
    args = parser.parse_args()    
    kwargs = vars(args)    
    main(**kwargs)
