import numpy as np
from . import AnnSeqContainer
from . import AnnSequence
from . import InvalidStrandType
from . import DictValidator
from . import Creator
from . import NotPositiveException
from . import UscuSeqConverter
from . import AnnSeqProcessor
class AnnChromCreator(Creator):
    """Map annotated sequences belong to specific chromosome on the chromosome"""
    def __init__(self):
        super().__init__()
        self._ann_seq_processor = AnnSeqProcessor()
    def _validate(self):
        pass
    def create(self,ann_seqs,chrom_id,length,source):
        self._validate()
        for ann_seq in ann_seqs:
            if str(chrom_id)!=str(ann_seq.chromosome_id):
                err = "Chromosome id and sequence id are not the same"
                raise Exception(err)
        ann_types = ann_seqs.ANN_TYPES
        chrom = self._get_init_chrom(chrom_id,ann_types,length,source)
        self._add_seqs(chrom,ann_seqs,length,source)
        return chrom
    def _get_init_chrom(self,chrom_id,ann_types,length,source):
        """Get initialized chromosome"""      
        chrom = AnnSeqContainer()
        chrom.ANN_TYPES = ann_types
        for strand in ['plus','minus']:
            ann_seq = AnnSequence()
            ann_seq.length = length
            ann_seq.chromosome_id = str(chrom_id)
            ann_seq.strand = strand
            ann_seq.source = source
            ann_seq.id = chrom_id+"_"+strand
            ann_seq.ANN_TYPES = ann_types
            ann_seq.init_space()
            chrom.add(ann_seq)
        return chrom
    def _add_seqs(self,chrom,ann_seqs,length,source):
        for ann_seq in ann_seqs:
            one_strand_chrom = chrom.get(str(ann_seq.chromosome_id)+"_"+ann_seq.strand)
            self._add_seq(one_strand_chrom,ann_seq,length,source)
    def _add_seq(self,one_strand_chrom,ann_seq,chrom_length,source):
        txStart = ann_seq.absolute_index
        txEnd = ann_seq.absolute_index+ann_seq.length - 1
        strand = ann_seq.strand
        if strand == 'plus':
            gene_start_index = txStart
            gene_end_index = txEnd
        elif strand == 'minus':
            gene_start_index = chrom_length-1-txEnd
            gene_end_index = chrom_length-1-txStart
        else:
            raise InvalidStrandType(strand)
        ann_seq.source = source
        for type_ in ann_seq.ANN_TYPES:
            seq = ann_seq.get_ann(type_)
            if strand=='minus':
                seq = np.flip(seq, 0)
            one_strand_chrom.add_ann(type_,seq,gene_start_index,gene_end_index)
class AnnGenomeCreator(Creator):
    """Map annotated sequences on the genome"""
    def __init__(self):
        super().__init__()
        self._chrom_creator = AnnChromCreator()
    def _validate(self):
        pass
    def create(self,ann_seqs,genome_information):
        validator = DictValidator(genome_information,['source','chromosome'],[],[])
        validator.validate()
        seqs_collect = {}
        genome = AnnSeqContainer()
        ann_types = list(ann_seqs.ANN_TYPES)
        genome.ANN_TYPES = ann_types
        for chrom_id in genome_information['chromosome'].keys():
            container = AnnSeqContainer()
            container.ANN_TYPES = ann_types
            seqs_collect[chrom_id] = container
        for ann_seq in ann_seqs:
            chrom_id = str(ann_seq.chromosome_id)
            seqs_collect[chrom_id].add(ann_seq)
        source = genome_information['source']
        for chrom_id,length in genome_information['chromosome'].items():
            ann_seqs = seqs_collect[chrom_id]
            chrom = self._chrom_creator.create(ann_seqs,chrom_id,length,source)
            genome.add(chrom)
        return genome
