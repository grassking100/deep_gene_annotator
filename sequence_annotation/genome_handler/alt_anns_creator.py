from ..utils.utils import get_gff_with_seq_id
from .exception import NotOneHotException
from .ann_seq_processor import is_one_hot
from .sequence import AnnSequence
from .seq_container import AnnSeqContainer
from .ann_genome_processor import get_backgrounded_genome

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
        ann_seq.strand = 'plus' if strand == '+' else 'minus'
        ann_seq.init_space()
        return ann_seq

    def _validate(self,ann_seq):
        if not is_one_hot(ann_seq,['exon','intron','alt_donor','alt_accept','other']):
            raise NotOneHotException(ann_seq.id)

    def convert(self,gff,genome_info):
        genome = AnnSeqContainer()
        genome.ANN_TYPES = self.ANN_TYPES
        for chrom_id,strand in zip(list(gff['chr']),list(gff['strand'])):
            if chrom_id not in genome.ids:
                chrom = self._create_seq(chrom_id,strand,genome_info[str(chrom_id)])
                genome.add(chrom)
        gff = get_gff_with_seq_id(gff)
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