from abc import abstractmethod
import numpy as np
from .sequence import AnnSequence, PLUS,STRANDS
from .ann_seq_processor import is_one_hot
from .exception import NotOneHotException, InvalidStrandType, NotSameSizeException

class AnnSeqConverter:
    """Convert zero-based dicitonary data into AnnSequence"""
    def __init__(self):
        self._ANN_TYPES = None
    
    @property
    def ANN_TYPES(self):
        return self._ANN_TYPES

    def _create_seq(self,name,chrom_id,strand,tx_start,tx_end):
        """Create annotation sequence for returning"""
        length = tx_end-tx_start+1
        ann_seq = AnnSequence()
        ann_seq.absolute_index = tx_start
        ann_seq.id = name
        ann_seq.length = length
        ann_seq.ANN_TYPES = self.ANN_TYPES
        ann_seq.chromosome_id = chrom_id
        ann_seq.strand = strand
        ann_seq.init_space()
        return ann_seq

    @abstractmethod
    def _validate(self,ann_seq):
        pass

    @abstractmethod
    def _create_template_ann_seq(self,length):
        """Create template annotation sequence"""

    @abstractmethod
    def convert(self,data):
        pass

class _GeneticSeqConverter(AnnSeqConverter):
    """Convert zero-based dicitonary data into AnnSequence with exon and intron type"""
    def __init__(self):
        super().__init__()
        self._ANN_TYPES = ['exon','intron']

    def _create_template_ann_seq(self,length):
        ann_seq = AnnSequence()
        ann_seq.length = length
        ann_seq.strand = PLUS
        ann_seq.ANN_TYPES = set(self.ANN_TYPES + ['gene'])
        ann_seq.init_space()
        return ann_seq

    def _validate(self,ann_seq):
        if not is_one_hot(ann_seq,self._ANN_TYPES):
            raise NotOneHotException(ann_seq.id)

class _CodingSeqConverter(AnnSeqConverter):
    """Convert zero-based dicitonary data into AnnSequence with coding annotation"""
    def __init__(self):
        super().__init__()
        self._ANN_TYPES = ['cds','intron','utr_5','utr_3']

    def _create_template_ann_seq(self,length):
        ann_seq = AnnSequence()
        ann_seq.length = length
        ann_seq.strand = PLUS
        extra_type = ['exon','ORF','utr','utr_5_potential','utr_3_potential','gene']
        ann_seq.ANN_TYPES = set(self.ANN_TYPES + extra_type)
        ann_seq.init_space()
        return ann_seq

    def _validate(self,ann_seq):
        if not is_one_hot(ann_seq,self._ANN_TYPES):
            raise NotOneHotException(ann_seq.id)

    def _validate_exon_status(self,ann_seq):
        added_exon = np.array([0]*ann_seq.length,dtype='float64')
        for type_ in ['utr_5','utr_3','cds']:
            added_exon += ann_seq.get_ann(type_)
        if not np.all(added_exon==ann_seq.get_ann('exon')):
            raise Exception("Exon status is not consistent with UTR and CDS")

class GeneticBedSeqConverter(_GeneticSeqConverter):
    def convert(self,data):
        """Convert zero-based dicitonary data into AnnSequence"""
        tx_start = data['start']
        tx_end = data['end']
        #Create template
        length = tx_end-tx_start+1
        template_ann_seq = self._create_template_ann_seq(length)
        #Annotation
        template_ann_seq.set_ann("gene",1)
        if len(data['block_related_start']) != len(data['block_related_end']):
            raise NotSameSizeException("block_related_star","block_related_end")

        for start,end in zip(data['block_related_start'],data['block_related_end']):
            template_ann_seq.add_ann('exon',1,start,end)

        template_ann_seq.op_not_ann("intron","gene","exon")
        ann_seq = self._create_seq(data['id'],data['chr'],data['strand'],tx_start,tx_end)

        for type_ in ann_seq.ANN_TYPES:
            data = template_ann_seq.get_ann(type_)
            ann_seq.add_ann(type_,data)
        self._validate(ann_seq)
        return ann_seq
            
class CodingBedSeqConverter(_CodingSeqConverter):
    def convert(self,data):
        """Convert zero-based dicitonary data into AnnSequence"""
        tx_start = data['start']
        tx_end = data['end']
        gene = GeneticBedSeqConverter().convert(data)
        length = gene.length
        template_ann_seq = self._create_template_ann_seq(gene.length)
        cds_start = data['thick_start'] - data['start']
        cds_end = data['thick_end'] - data['start']
        if cds_start <= cds_end:
            template_ann_seq.set_ann('ORF', 1 ,cds_start,cds_end)
        template_ann_seq.set_ann('exon',gene.get_ann('exon'))
        template_ann_seq.set_ann('intron',gene.get_ann('intron'))
        template_ann_seq.op_and_ann('cds','exon','ORF')
        template_ann_seq.op_not_ann('utr','exon','ORF')
        utr = template_ann_seq.get_ann('utr')
        if cds_start <= cds_end:
            if gene.strand not in STRANDS:
                raise InvalidStrandType(gene.strand)

            if gene.strand == PLUS:
                if cds_start-1 >= 0:
                    template_ann_seq.set_ann('utr_5_potential',1,0,cds_start-1)
                if cds_end+1 <= length-1:
                    template_ann_seq.set_ann('utr_3_potential',1,cds_end+1,length-1)
            else:
                if cds_start-1 >= 0:
                    template_ann_seq.set_ann('utr_3_potential',1,0,cds_start-1)
                if cds_end+1 <= length-1:
                    template_ann_seq.set_ann('utr_5_potential',1,cds_end+1,length-1)

            template_ann_seq.op_and_ann('utr_5','utr_5_potential','utr')
            template_ann_seq.op_and_ann('utr_3','utr_3_potential','utr')
        else:
            template_ann_seq.set_ann('utr_5',utr)

        self._validate_exon_status(template_ann_seq)
        ann_seq = self._create_seq(gene.id,gene.chromosome_id,gene.strand,tx_start,tx_end)
        for type_ in ann_seq.ANN_TYPES:
            data = template_ann_seq.get_ann(type_)
            ann_seq.set_ann(type_,data)
        self._validate(ann_seq)
        return ann_seq
    