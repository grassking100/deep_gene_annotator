import os, sys
import deepdish as dd
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.file_process.utils import read_gff,get_gff_with_intron,PLUS
from sequence_annotation.file_process.utils import EXON_SKIPPING,INTRON_RETENTION,EXON_TYPE,INTRON_TYPE
from sequence_annotation.file_process.utils import ALT_STATUSES, BASIC_GFF_FEATURES
from sequence_annotation.file_process.utils import INTERGENIC_REGION_TYPE, TRANSCRIPT_TYPE
from sequence_annotation.file_process.get_region_table import read_region_table
from sequence_annotation.genome_handler.exception import NotOneHotException
from sequence_annotation.genome_handler.ann_seq_processor import is_one_hot
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.ann_genome_processor import get_backgrounded_genome


class Gff2AnnSeqs:
    def __init__(self,ann_types):
        self._ANN_TYPES = ann_types
    
    @property
    def ANN_TYPES(self):
        return self._ANN_TYPES

    def _create_seq(self,chrom_id,strand,length):
        """Create annotation sequence for returning"""
        ann_seq = AnnSequence()
        ann_seq.id = chrom_id
        ann_seq.length = length
        ann_seq.strand = strand
        ann_seq.ANN_TYPES = self.ANN_TYPES
        ann_seq.chromosome_id = chrom_id
        ann_seq.init_space()
        return ann_seq

    def _validate(self,ann_seq):
        if not is_one_hot(ann_seq,[EXON_TYPE,INTRON_TYPE,INTERGENIC_REGION_TYPE]):
            raise NotOneHotException(ann_seq.id)

    def convert(self,gff,region_table,source):
        if set(gff['strand']) != set(['+']):
            raise
        gff = get_gff_with_intron(gff)
        transcripts = gff[gff['feature']==TRANSCRIPT_TYPE].to_dict('record')
        blocks = gff[gff['feature']!=TRANSCRIPT_TYPE].groupby('parent')
        genome = AnnSeqContainer()
        genome.ANN_TYPES = self.ANN_TYPES
        for region in region_table.to_dict('record'):
            chrom = self._create_seq(region['ordinal_id_with_strand'],PLUS,region['length'])
            chrom.source = source
            genome.add(chrom)

        for transcript in transcripts:
            blocks_ = blocks.get_group(transcript['id']).to_dict('record')
            for block in blocks_:
                if block['feature'] in self.ANN_TYPES:
                    chrom = genome.get(transcript['chr'])
                    chrom.set_ann(block['feature'],1,block['start']-1,block['end']-1)
                    
        frontgrounded_type = list(self._ANN_TYPES)
        frontgrounded_type.remove(INTERGENIC_REGION_TYPE)
        backgrounded = get_backgrounded_genome(genome,INTERGENIC_REGION_TYPE,frontgrounded_type)
        for chrom in backgrounded:
            self._validate(chrom)
        return backgrounded

    
def create_ann_genome(gff,region_table,souce_name,ann_types):
    converter = Gff2AnnSeqs(ann_types)
    genome = converter.convert(gff,region_table,souce_name)   
    return genome


def main(gff_path,region_table_path,souce_name,output_path,
         with_alt_region=False):
    region_table = read_region_table(region_table_path)
    gff = read_gff(gff_path,with_attr=True,valid_features=BASIC_GFF_FEATURES+ALT_STATUSES)
    ann_types = [EXON_TYPE,INTRON_TYPE,INTERGENIC_REGION_TYPE]
    if with_alt_region:
        ann_types += [EXON_SKIPPING,INTRON_RETENTION]
    genome = create_ann_genome(gff,region_table,souce_name,ann_types)
    try:
        dd.io.save(output_path,genome.to_dict())
    except OverflowError:
        import pickle
        with open(output_path, 'wb') as fp:
            pickle.dump(genome.to_dict(), fp)
            
