import os, sys
import deepdish as dd
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import get_gff_with_attribute, read_gff, read_bed
from sequence_annotation.genome_handler.exception import NotOneHotException
from sequence_annotation.genome_handler.ann_seq_processor import is_one_hot
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.ann_genome_processor import get_backgrounded_genome
from sequence_annotation.gene_info.utils import GENE_TYPES

strand_convert = {'+':'plus','-':'minus'}

class Gff2AnnSeqs:
    def __init__(self):
        self._ANN_TYPES = ['alt_accept','alt_donor','exon_skipping',
                           'intron_retetion','exon','intron','other']
    
    @property
    def ANN_TYPES(self):
        return self._ANN_TYPES

    def _create_seq(self,chrom_id,strand,length):
        """Create annotation sequence for returning"""
        ann_seq = AnnSequence()
        ann_seq.id = chrom_id
        ann_seq.length = length
        ann_seq.strand = strand_convert[strand]
        ann_seq.ANN_TYPES = self.ANN_TYPES
        ann_seq.chromosome_id = chrom_id
        ann_seq.init_space()
        return ann_seq

    def _validate(self,ann_seq):
        if not is_one_hot(ann_seq,['exon','intron','alt_donor','alt_accept','other']):
            raise NotOneHotException(ann_seq.id)

    def convert(self,gff,region_bed,source):
        if 'parent' not in gff.columns:
            raise Exception('Parent should be in gff data')
        gff = get_gff_with_attribute(gff)
        is_gene = gff['feature'].isin(GENE_TYPES)
        genes = gff[is_gene]
        genome = AnnSeqContainer()
        genome.ANN_TYPES = self.ANN_TYPES
        for region in region_bed.to_dict('record'):
            chrom = self._create_seq(region['id'],region['strand'],region['end']-region['start']+1)
            chrom.source = source
            genome.add(chrom)
        genes = genes.to_dict('record')
        blocks = gff[~is_gene].groupby('parent')
        for gene in genes:
            id_ = gene['id']
            blocks_ = blocks.get_group(id_).to_dict('record')
            for block in blocks_:
                if block['feature'] in self.ANN_TYPES:
                    chrom = genome.get(gene['chr'])
                    chrom.set_ann(block['feature'],1,block['start']-1,block['end']-1)
        backgrounded = get_backgrounded_genome(genome,'other',['exon','intron','alt_donor','alt_accept'])
        for chrom in backgrounded:
            self._validate(chrom)
        return backgrounded

def alt_anns_creator(gff,region_bed,souce_name):
    gff = get_gff_with_attribute(gff)
    converter = Gff2AnnSeqs()
    genome = converter.convert(gff,region_bed,souce_name)   
    return genome

def main(gff_path,region_bed_path,souce_name,output_path):
    gff = read_gff(gff_path)
    region_bed = read_bed(region_bed_path)
    converter = Gff2AnnSeqs()
    genome = alt_anns_creator(gff,region_bed,souce_name)
    try:
        dd.io.save(output_path,genome.to_dict())
    except OverflowError:
        import pickle
        with open(output_path, 'wb') as fp:
            pickle.dump(genome.to_dict(), fp)
            
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--gff_path",required=True)
    parser.add_argument("-r", "--region_bed_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("-s", "--souce_name",required=True)
    args = parser.parse_args()
    main(args.gff_path,args.region_bed_path,args.souce_name,args.output_path)