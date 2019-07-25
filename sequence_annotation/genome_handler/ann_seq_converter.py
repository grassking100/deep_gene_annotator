from abc import ABCMeta,abstractmethod,abstractproperty
import numpy as np
from .sequence import AnnSequence
from .ann_seq_processor import is_one_hot
from .exception import NotOneHotException,InvalidStrandType
from ..utils.exception import NotPositiveException

class AnnSeqConverter:
    """Converter to convert zero-based dicitonary data into AnnSequence"""
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
        pass

    @abstractmethod
    def convert(self,data):
        pass

class _GeneticSeqConverter(AnnSeqConverter):
    """Converter to convert zero-based dicitonary data into AnnSequence with exon and intron type"""
    def __init__(self):
        super().__init__()
        self._ANN_TYPES = ['exon','intron']

    def _create_template_ann_seq(self,length):
        ann_seq = AnnSequence()
        ann_seq.length = length
        ann_seq.strand = 'plus'
        #ann_seq.id = 'template'
        ann_seq.ANN_TYPES = set(self.ANN_TYPES + ['gene'])
        ann_seq.init_space()
        return ann_seq

    def _validate(self,ann_seq):
        if not is_one_hot(ann_seq,self._ANN_TYPES):
            raise NotOneHotException(ann_seq.id)

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
        for start,end in zip(data['block_related_start'],data['block_related_end']):
            template_ann_seq.add_ann('exon',1,start,end)

        template_ann_seq.op_not_ann("intron","gene","exon")
        ann_seq = self._create_seq(data['id'],data['chr'],data['strand'],tx_start,tx_end)

        for type_ in ann_seq.ANN_TYPES:
            data = template_ann_seq.get_ann(type_)
            ann_seq.add_ann(type_,data)
        self._validate(ann_seq)
        return ann_seq

class _CodingSeqConverter(AnnSeqConverter):
    """Converter to convert zero-based dicitonary data into AnnSequence with coding annotation"""
    def __init__(self):
        super().__init__()
        self._ANN_TYPES = ['cds','intron','utr_5','utr_3']

    def _create_template_ann_seq(self,length):
        ann_seq = AnnSequence()
        ann_seq.length = length
        ann_seq.strand = 'plus'
        #ann_seq.id = 'template'
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
        utr = template_ann_seq.get_ann('utr')
        if np.any(utr==1):
            if  gene.strand == 'plus':
                if cds_start-1 >= 0:
                    template_ann_seq.set_ann('utr_5_potential',1,0,cds_start-1)
                if cds_end+1 <= length-1:
                    template_ann_seq.set_ann('utr_3_potential',1,cds_end+1,length-1)
            elif gene.strand == 'minus':
                if cds_start-1 >= 0:
                    template_ann_seq.set_ann('utr_3_potential',1,0,cds_start-1)
                if cds_end+1 <= length-1:
                    template_ann_seq.set_ann('utr_5_potential',1,cds_end+1,length-1)
            else:
                raise InvalidStrandType(gene.strand)
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

class UCSCSeqConverter(_CodingSeqConverter):
    """Converter to convert zero-based UCSC data into AnnSequence with coding annotation"""
    def convert(self,data):
        """Convert zero-based dicitonary data into AnnSequence"""
        tx_start = data['txStart']
        tx_end = data['txEnd']
        #calculate for template container
        cds_start=data['cdsStart'] - tx_start
        cds_end=data['cdsEnd'] - tx_start
        length = tx_end-tx_start+1
        strand = data['strand']
        template_ann_seq = self._create_template_ann_seq(length)
        template_ann_seq.set_ann('gene',1)
        if data['exonCount'] <= 0:
            raise NotPositiveException("exonCount",data['exonCount'])
        else:
            exonStarts = [start_index - tx_start for start_index in data['exonStarts']]
            exonEnds = [end_index - tx_start for end_index in data['exonEnds']]
        #if exon exist, it will add cds, utr (if exists), intron (if exists) in template container.
        for exon_index in range(data['exonCount']):
            start = exonStarts[exon_index]
            end = exonEnds[exon_index]
            template_ann_seq.set_ann('exon', 1, start, end)
        if cds_start <= cds_end:
            template_ann_seq.set_ann('ORF', 1 ,cds_start,cds_end)
        template_ann_seq.op_and_ann('cds','exon','ORF')
        template_ann_seq.op_not_ann('utr','exon','ORF')
        template_ann_seq.op_not_ann('intron','gene','exon')
        utr = template_ann_seq.get_ann('utr')
        if np.any(utr==1):
            if np.any(template_ann_seq.get_ann('cds')==1):
                if  strand== 'plus':
                    if cds_start-1 >= 0:
                        template_ann_seq.set_ann('utr_5_potential',1,0,cds_start-1)
                    if cds_end+1 <= length-1:
                        template_ann_seq.set_ann('utr_3_potential',1,cds_end+1,length-1)
                elif strand == 'minus':
                    if cds_start-1 >= 0:
                        template_ann_seq.set_ann('utr_3_potential',1,0,cds_start-1)
                    if cds_end+1 <= length-1:
                        template_ann_seq.set_ann('utr_5_potential',1,cds_end+1,length-1)
                else:
                    raise InvalidStrandType(strand)
                template_ann_seq.op_and_ann('utr_5','utr_5_potential','utr')
                template_ann_seq.op_and_ann('utr_3','utr_3_potential','utr')
            else:
                template_ann_seq.set_ann('utr_5',utr)
        #add template annotated chromosmome to whole annotated chromosmome
        name = data['name']
        chrom_id = data['chrom']
        self._validate_exon_status(template_ann_seq)
        ann_seq = self._create_seq(name,chrom_id,strand,tx_start,tx_end)
        for type_ in ann_seq.ANN_TYPES:
            data = template_ann_seq.get_ann(type_)
            ann_seq.set_ann(type_,data)
        self._validate(ann_seq)
        return ann_seq

class EnsemblSeqConverter(_CodingSeqConverter):
    """Converter to convert zero-based Ensembl data into AnnSequence with coding annotation"""
    def convert(self,data):
        #Convert zero-based dicitonary data into AnnSequence
        tx_start = data['tx_start']
        tx_end = data['tx_end']
        #Create template
        length = tx_end-tx_start+1
        template_ann_seq = self._create_template_ann_seq(length)
        #Set Annotation
        template_ann_seq.set_ann("gene",1)
        site_info = {'exon':(data['exons_start'],data['exons_end']),
                     'cds':(data['cdss_start'],data['cdss_end']),
                     'utr_5':(data['utrs_5_start'],data['utrs_5_end']),
                     'utr_3':(data['utrs_3_start'],data['utrs_3_end'])}
        for type_,sites in site_info.items():
            starts,ends = sites
            if len(starts) != len(ends):
                raise Exception("Start and end sites number is not same")
            for (start,end) in zip(starts,ends):
                template_ann_seq.add_ann(type_,1,start-tx_start,end-tx_start)
        template_ann_seq.op_not_ann("intron","gene","exon")
        self._validate_exon_status(template_ann_seq)
        ann_seq = self._create_seq(data['protein_id'],data['chrom'],
                                   data['strand'],tx_start,tx_end)
        for type_ in ann_seq.ANN_TYPES:
            data = template_ann_seq.get_ann(type_)
            ann_seq.add_ann(type_,data)
        self._validate(ann_seq)
        return ann_seq
    