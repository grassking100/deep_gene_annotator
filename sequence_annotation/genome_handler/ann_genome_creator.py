import numpy as np
from . import AnnSeqContainer
from . import AnnSequence
from . import InvalidStrandType
from . import DictValidator
from . import Creator
from . import NotPositiveException
from . import UscuSeqConverter
class AnnGenomeCreator(Creator):
    """Purpose:Make sequences of numeric value represent annotated region occupy or not"""
    def __init__(self,genome_information,data,converter=UscuSeqConverter):
        super().__init__()
        self._data_information = genome_information
        self._data = data
        self._converter = converter()
        self._ANN_TYPES = self._converter.ANN_TYPES+['other']
    def _validate(self):
        pass
    def create(self):
        self._validate()
        self._validate_genome_info()
        self._result = self._get_init_genome()
        self._annotate_genome()
        return self._result
    def _validate_genome_info(self):
        validator = DictValidator(self._data_information,['source','chromosome'],[],[])
        validator.validate()
    def _get_init_genome(self):
        """Get initialized genome"""        
        genome = AnnSeqContainer()
        genome.ANN_TYPES = self._ANN_TYPES
        for id_, length in self._data_information['chromosome'].items():
            for strand in ['plus','minus']:
                ann_seq = AnnSequence()
                ann_seq.length = length
                ann_seq.chromosome_id = str(id_)
                ann_seq.strand = strand
                ann_seq.source = self._data_information['source']
                ann_seq.id = id_+"_"+strand
                ann_seq.ANN_TYPES = self._ANN_TYPES
                ann_seq.init_space()
                genome.add(ann_seq)
        return genome
    def _add_seq_to_genome(self, ann_seq, start_index, end_index):
        chrom_id = ann_seq.chromosome_id
        strand = ann_seq.strand
        chrom = self._result.get(chrom_id+"_"+strand)
        for type_ in ann_seq.ANN_TYPES:
            seq = ann_seq.get_ann(type_)
            if strand=='minus':
                seq = np.flip(seq, 0)
            chrom.add_ann(type_,seq,start_index,end_index)
    def _annotate_genome(self):
        """Annotate genome"""
        for seq_data in self._data:
            self._annotate_seq(seq_data)
    def _annotate_seq(self,seq_data):
        txStart = seq_data['txStart']
        txEnd = seq_data['txEnd']
        #calculate for template container
        strand = seq_data['strand']
        chrom_length = self._data_information['chromosome'][seq_data['chrom']]
        if strand== 'plus':
            gene_start_index = txStart
            gene_end_index = txEnd
        elif strand == 'minus':
            gene_start_index = chrom_length-1-txEnd
            gene_end_index = chrom_length-1-txStart
        else:
            raise InvalidStrandType(strand)
        ann_seq = self._converter.convert(seq_data)
        ann_seq.source = self._data_information['source']
        self._add_seq_to_genome(ann_seq,gene_start_index,gene_end_index)
