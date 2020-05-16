import os
import sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Pool
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import BASIC_GENE_MAP, get_gff_with_attribute
from sequence_annotation.utils.utils import create_folder, write_gff, write_json, read_json, read_gff, read_fasta,write_bed
from sequence_annotation.genome_handler.ann_seq_processor import get_background
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.region_extractor import RegionExtractor,GeneInfoExtractor
from sequence_annotation.genome_handler.sequence import AnnSequence, PLUS
from sequence_annotation.preprocess.utils import RNA_TYPES, get_gff_with_intron, get_gff_with_intergenic_region, read_region_table
from sequence_annotation.preprocess.gff2bed import gff2bed
from sequence_annotation.process.flip_and_rename_coordinate import flip_and_rename_gff
from sequence_annotation.postprocess.boundary_process import get_valid_intron_boundary, get_splice_pairs, find_substr
from sequence_annotation.postprocess.boundary_process import get_splice_pair_statuses, get_exon_boundary


def _create_ann_seq(chrom, length, ann_types):
    ann_seq = AnnSequence(ann_types, length)
    ann_seq.id = ann_seq.chromosome_id = chrom
    ann_seq.strand = PLUS
    return ann_seq


def remove_block(dict_,block):
    del dict_[block.start]
    if block.end in dict_:
        del dict_[block.end]
    
def set_block(dict_,block):
    dict_[block.start] = block
    dict_[block.end] = block

class GFFReviser:
    def __init__(self, gene_info_extractor, donor_pattern=None, acceptor_pattern=None,
                 length_thresholds=None, donor_distance=None, acceptor_distance=None,
                 donor_index_shift=None, acceptor_index_shift=None,methods=None,**kwargs):
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
        self._gene_info_extractor = gene_info_extractor
        self._extractor = RegionExtractor()
        self.donor_pattern = donor_pattern or 'GT'
        self.acceptor_pattern = acceptor_pattern or 'AG'
        if length_thresholds is not None:
            if set(length_thresholds.keys()) != set(
                    ['exon', 'intron', 'other', 'transcript']):
                raise Exception("Wrong key in length_thresholds")
        self.length_thresholds = length_thresholds
        self.donor_distance = donor_distance or 0
        self.acceptor_distance = acceptor_distance or 0
        self.donor_index_shift = donor_index_shift or 0
        self.acceptor_index_shift = acceptor_index_shift or 1
        self._ann_types = ['exon', 'intron', 'other']
        self.methods = methods or []
        
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['gene_info_extractor'] = self._gene_info_extractor.get_config()
        config['donor_pattern'] = self.donor_pattern
        config['acceptor_pattern'] = self.acceptor_pattern
        config['length_thresholds'] = self.length_thresholds
        config['donor_distance'] = self.donor_distance
        config['acceptor_distance'] = self.acceptor_distance
        config['ann_types'] = self._ann_types
        config['donor_index_shift'] = self.donor_index_shift
        config['acceptor_index_shift'] = self.acceptor_index_shift
        config['methods'] = self.methods
        return config

    def _validate_gff(self, gff):
        strand = list(set(gff['strand']))
        if len(strand) != 1 or strand[0] != '+':
            raise Exception("Got strands, {}".format(strand))

    def _create_ann_seq(self, chrom, length, gff):
        ann_seq = _create_ann_seq(chrom, length, self._ann_types)
        exon_introns = gff[gff['feature'].isin(
            ['exon', 'intron'])].to_dict('record')
        for item in exon_introns:
            ann_seq.add_ann(item['feature'],1,
                            item['start'] - 1,
                            item['end'] - 1)
        other = get_background(ann_seq, ['exon', 'intron'])
        ann_seq.add_ann('other', other)
        return ann_seq

    def _merge_current_to_previous(self,previous_block,current_block,block_table,fragment_lengths):
        remove_block(block_table,current_block)
        remove_block(block_table,previous_block)
        del fragment_lengths[current_block.start]
        if previous_block.start in fragment_lengths:
            del fragment_lengths[previous_block.start]
        previous_block.end = current_block.end
        #If merged block is fragment, then update its length in fragment_lengths
        if previous_block.length < self.length_thresholds[previous_block.ann_type]:
            fragment_lengths[previous_block.start] = previous_block.length
        set_block(block_table,previous_block)
            
    def _merge_current_to_next(self,current_block,next_block,block_table,fragment_lengths):
        remove_block(block_table,current_block)
        remove_block(block_table,next_block)
        del fragment_lengths[current_block.start]
        if next_block.start in fragment_lengths:
            del fragment_lengths[next_block.start]
        next_block.start = current_block.start
        #If merged block is fragment, then update its length in fragment_lengths
        if next_block.length < self.length_thresholds[next_block.ann_type]:
            fragment_lengths[next_block.start] = next_block.length
        set_block(block_table,next_block)

    def _merge_blocks_to_previous(self,previous_block,current_block,next_block,block_table,fragment_lengths):
        remove_block(block_table,current_block)
        remove_block(block_table,previous_block)
        remove_block(block_table,next_block)
        del fragment_lengths[current_block.start]
        if next_block.start in fragment_lengths:
            del fragment_lengths[next_block.start]
        if previous_block.start in fragment_lengths:
            del fragment_lengths[previous_block.start]
        previous_block.end = next_block.end
        #If merged block is fragment, then update its length in fragment_lengths
        if previous_block.length < self.length_thresholds[previous_block.ann_type]:
            fragment_lengths[previous_block.start] = previous_block.length
        #If merged block is not fragment and is in fragment_lengths, then delete it form fragment_lengths
        set_block(block_table,previous_block)
    
    def _filter_ann_by_length(self, seq_info_container):
        if self.length_thresholds is None:
            return seq_info_container
        block_table = {}
        fragment_lengths = {}
        for seq_info in seq_info_container:
            block_table[seq_info.start] = seq_info
            block_table[seq_info.end] = seq_info
            if seq_info.length < self.length_thresholds[seq_info.ann_type]:
                fragment_lengths[seq_info.start] = seq_info.length
        
        while len(fragment_lengths)>0:
            starts = list(fragment_lengths.keys())
            fragment_lengths_ = list(fragment_lengths.values())
            index = np.argsort(fragment_lengths_)[0]
            start = starts[index]
            current_block = block_table[start]
            previous_block = next_block = None
            if current_block.start-1 in block_table:
                previous_block = block_table[current_block.start-1]
            if current_block.end+1 in block_table:
                next_block = block_table[current_block.end+1]
            if previous_block is not None and next_block is not None:
                #If neighbors have same type, then merge three blocks to one blocks
                if previous_block.ann_type == next_block.ann_type:
                    self._merge_blocks_to_previous(previous_block,current_block,next_block,block_table,fragment_lengths)
                else:
                    #Merge current block to larger block
                    if previous_block.length >= next_block.length:
                        self._merge_current_to_previous(previous_block,current_block,block_table,fragment_lengths)
                    else:
                        self._merge_current_to_next(current_block,next_block,block_table,fragment_lengths)
            #Merge current block to previous block
            elif previous_block is not None:
                self._merge_current_to_previous(previous_block,current_block,block_table,fragment_lengths)
            #Merge current block to previous next
            else:
                self._merge_current_to_next(current_block,next_block,block_table,fragment_lengths)

        returned = SeqInfoContainer()
        returned.note = seq_info_container.note
        for block in block_table.values():
            if block.id not in returned.ids:
                returned.add(block)
        
        return returned

    def _process_transcript(self, rna, splice_pair_statuses, ann_seq):
        rna_boundary = rna['start'], rna['end']
        intron_boundarys = get_valid_intron_boundary(
            rna_boundary, splice_pair_statuses)
        rna_exon_boundarys = get_exon_boundary(rna_boundary, intron_boundarys)
        for boundary in rna_exon_boundarys:
            ann_seq.add_ann('exon', 1, boundary[0] - 1, boundary[1] - 1)
        for boundary in intron_boundarys:
            ann_seq.add_ann('intron', 1, boundary[0] - 1, boundary[1] - 1)

    def _preprocess_gff(self, chrom_id, length, gff):
        self._validate_gff(gff)
        gff = gff[gff['chr'] == chrom_id]
        gff = get_gff_with_attribute(gff)
        if 'intron' not in set(gff['feature']):
            gff = get_gff_with_intron(gff)
        if 'intergenic region' not in set(gff['feature']):
            region_table = pd.DataFrame.from_dict(
                [{'chrom_id': chrom_id, 'length': length, 'strand': '+'}])
            gff = get_gff_with_intergenic_region(gff, region_table, 'chrom_id')
        gff.loc[gff['feature'] == 'intergenic region', 'feature'] = 'other'
        return gff

    def filter_by_threshold(self, chrom_id, length, gff):
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
        # Filter exon,intron,other
        gff = self._preprocess_gff(chrom_id, length, gff)
        ann_seq = self._create_ann_seq(chrom_id, length, gff)
        blocks = self._extractor.extract(ann_seq)
        gff = self._filter_ann_by_length(blocks).to_gff()
        ann_seq = self._create_ann_seq(chrom_id, length, gff)
        filtered = self._gene_info_extractor.extract_per_seq(ann_seq)
        filtered_gff = None
        if len(filtered) > 0:
            filtered_gff = filtered.to_gff()
            if self.length_thresholds is not None:
                # Filter transcript
                invalid_ids = []
                filtered_gff = get_gff_with_attribute(filtered_gff)
                transcripts = filtered_gff[filtered_gff['feature'].isin(
                    RNA_TYPES)].to_dict('record')
                for transcript in transcripts:
                    length = transcript['end'] - transcript['start'] + 1
                    if length < self.length_thresholds['transcript']:
                        invalid_ids.append(transcript['parent'])
                        invalid_ids.append(transcript['id'])

                filtered_gff = filtered_gff[(~filtered_gff['parent'].isin(
                    invalid_ids)) & (~filtered_gff['id'].isin(invalid_ids))]
        return filtered_gff

    def fix_splicing_site(self, chrom_id, length, seq, gff):
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
        gff = self._preprocess_gff(chrom_id, length, gff)
        splice_pairs = get_splice_pairs(gff)
        ann_donor_sites = find_substr(
            self.donor_pattern, seq, self.donor_index_shift + 1)
        ann_acceptor_sites = find_substr(
            self.acceptor_pattern, seq, self.acceptor_index_shift + 1)
        splice_pair_statuses = get_splice_pair_statuses(splice_pairs, ann_donor_sites, ann_acceptor_sites,
                                                        self.donor_distance, self.acceptor_distance)
        fixed_ann_seq = _create_ann_seq(chrom_id, length, self._ann_types)
        transcripts = gff[gff['feature'].isin(RNA_TYPES)].to_dict('record')
        for transcript in transcripts:
            self._process_transcript(
                transcript, splice_pair_statuses, fixed_ann_seq)
        other = get_background(fixed_ann_seq, ['exon', 'intron'])
        fixed_ann_seq.add_ann('other', other)
        info = self._gene_info_extractor.extract_per_seq(fixed_ann_seq)
        fixed_gff = None
        if len(info) > 0:
            fixed_gff = info.to_gff()
        return fixed_gff

    def process(self, chrom_id, length, seq, gff):
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
        gff = gff[gff['chr'] == chrom_id]
        for method in self.methods:
            if gff is None or len(gff) == 0:
                break
            if method == 'length_threshold':
                gff = self.filter_by_threshold(chrom_id, length, gff)
            elif method == 'distance_threshold':
                gff = self.fix_splicing_site(chrom_id, length, seq, gff)
            else:
                raise Exception("Invalid method {}".format(method))
        if gff is not None and len(gff)>0:
            gff = gff[gff['feature'].isin(['gene','mRNA','exon'])]
        return gff


def build_gff_reviser(simply_map, **kwargs):
    gene_info_extractor = GeneInfoExtractor(simply_map)
    gff_reviser = GFFReviser(gene_info_extractor, **kwargs)
    return gff_reviser

def _revise_and_save(reviser,chrom_id,length,seq,plus_strand_gff):
    print("Revising data {}".format(chrom_id))
    gff = reviser.process(chrom_id,length,seq,plus_strand_gff)
    return gff

def revise(plus_strand_gff, region_table, fasta, reviser,multiprocess=None):
    chroms = set(plus_strand_gff['chr'])
    kwarg_list = []
    lengths = dict(zip(region_table['ordinal_id_with_strand'] ,region_table['length']))
    groups = plus_strand_gff.groupby('chr')
    for chrom_id in chroms:
        length = lengths[chrom_id]
        seq = fasta[chrom_id]
        gff = groups.get_group(chrom_id)
        kwarg_list.append((reviser,chrom_id,length,seq,gff))
    if multiprocess is None:
        gff_list = [_revise_and_save(*kwargs) for kwargs in kwarg_list]
    else:
        with Pool(processes=multiprocess) as pool:
            gff_list = pool.starmap(_revise_and_save, kwarg_list)
    gff = pd.concat(gff_list, sort=True).reset_index(drop=True)
    return gff

def main(output_root, plus_strand_gff_path, region_table_path, fasta_path,
         revised_config_path,multiprocess=None):
    create_folder(output_root)
    revised_config = read_json(revised_config_path)
    del revised_config['class']
    del revised_config['gene_info_extractor']
    
    region_table = read_region_table(region_table_path)
    plus_strand_gff = read_gff(plus_strand_gff_path)
    fasta = read_fasta(fasta_path)
    reviser = build_gff_reviser(BASIC_GENE_MAP,**revised_config)
    config = reviser.get_config()
    config_path = os.path.join(output_root, "reviser_config.json")
    write_json(config, config_path)
    gff = revise(plus_strand_gff, region_table, fasta, reviser,multiprocess)
    plus_revised_gff_path = os.path.join(output_root, "revised_plus_strand.gff3")
    write_gff(gff, plus_revised_gff_path)
    flipped_gff = flip_and_rename_gff(gff,region_table)
    revised_gff_path = os.path.join(output_root, "revised_double_strand.gff3")
    write_gff(flipped_gff, revised_gff_path)
    revised_bed_path = os.path.join(output_root, "revised_double_strand.bed")
    flipped_bed = gff2bed(flipped_gff)
    write_bed(flipped_bed, revised_bed_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Revise the GFF')
    parser.add_argument("-i", "--plus_strand_gff_path", help="The path of origin "
                        "of single-strand plus-only data in GFf format", required=True)
    parser.add_argument("-t","--region_table_path",required=True,
                        help="The path of region data")
    parser.add_argument("-f","--fasta_path",required=True,
                        help="The path of fasta")
    parser.add_argument("-o","--output_root",required=True,
                        help="The root to save result")
    parser.add_argument("-c","--revised_config_path",required=True)
    parser.add_argument("--multiprocess", type=int,default=None)
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
