import os, sys
import deepdish as dd
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import get_gff_with_attribute, read_gff, read_fai
from sequence_annotation.genome_handler.exception import NotOneHotException
from sequence_annotation.genome_handler.ann_seq_processor import is_one_hot
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.ann_genome_processor import get_backgrounded_genome

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
        ann_seq.ANN_TYPES = self.ANN_TYPES
        ann_seq.chromosome_id = chrom_id
        ann_seq.strand = strand_convert[strand]
        ann_seq.init_space()
        return ann_seq

    def _validate(self,ann_seq):
        if not is_one_hot(ann_seq,['exon','intron','alt_donor','alt_accept','other']):
            raise NotOneHotException(ann_seq.id)

    def convert(self,gff,genome_info,source):
        if 'parent' not in gff.columns:
            raise Exception('Parent should be in gff data')
        genome = AnnSeqContainer()
        genome.ANN_TYPES = self.ANN_TYPES
        for chrom_id,strand in zip(list(gff['chr']),list(gff['strand'])):
            if chrom_id not in genome.ids:
                chrom = self._create_seq(chrom_id,strand,genome_info[str(chrom_id)])
                chrom.source = source
                genome.add(chrom)

        gff = get_gff_with_attribute(gff)
        genes = gff[gff['feature']=='gene'].to_dict('record')
        blocks = gff[gff['feature'] != 'gene'].groupby('parent')
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

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--gff_path",required=True)
    parser.add_argument("-f", "--fai_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("-s", "--souce_name",required=True)
    args = parser.parse_args()
    gff = get_gff_with_attribute(read_gff(args.gff_path))
    converter = Gff2AnnSeqs()
    fai = read_fai(args.fai_path)
    genome = converter.convert(gff,fai,args.souce_name)
    dd.io.save(args.output_path,genome.to_dict())