import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from .sequence import AnnSequence
from .ann_seq_processor import is_one_hot
from .exception import NotOneHotException

class AnnSeqConverter(metaclass=ABCMeta):
    def __init__(self,foucus_type,extra_types=None):
        self._foucus_type = foucus_type
        self._extra_types = extra_types or []
        self._ANN_TYPES = list(set(self._foucus_type).union(set(self._extra_types)))
    @property
    def ANN_TYPES(self):
        return self._ANN_TYPES
    def _create_seq(self,name,chrom_id,strand,tx_start,tx_end):
        length = tx_end-tx_start+1
        nt_number = length
        ann_seq = AnnSequence()
        ann_seq.absolute_index = tx_start
        ann_seq.id = name
        ann_seq.length = nt_number
        ann_seq.ANN_TYPES = self.ANN_TYPES
        ann_seq.chromosome_id = chrom_id
        ann_seq.strand = strand
        ann_seq.init_space()
        return ann_seq
    @abstractmethod
    def _validate(self,ann_seq,focus_types=None):
        pass
    @abstractmethod
    def _get_template_ann_seq(self,length):
        pass
    @abstractmethod
    def convert(self,data):
        """Convert zero-based dicitonary data into AnnSequence"""
        pass

class GeneticAnnSeqConverter(AnnSeqConverter):
    def __init__(self,foucus_type=None,extra_types=None):
        foucus_type = foucus_type or ['exon','intron']
        extra_types = extra_types or []
        super().__init__(foucus_type,extra_types)
    def _get_template_ann_seq(self,length):
        """Get template container"""
        ann_seq = AnnSequence()
        ann_seq.length = length
        ann_seq.strand = 'plus'
        ann_seq.id = 'template'
        ann_seq.ANN_TYPES = set(self.ANN_TYPES + ['gene'])
        ann_seq.init_space()
        return ann_seq
    def _validate(self,ann_seq,focus_types=None):
        if not is_one_hot(ann_seq,focus_types):
            raise NotOneHotException(ann_seq.id)

class GeneticBedSeqConverter(GeneticAnnSeqConverter):
    def convert(self,data):
        """Convert zero-based dicitonary data into AnnSequence"""
        tx_start = data['chromStart']
        tx_end = data['chromEnd']
        """Create template"""
        length = tx_end-tx_start+1
        template_ann_seq = self._get_template_ann_seq(length)
        """Annotation"""
        template_ann_seq.set_ann("gene",1)
        site_info = {'exon':(data['blockStarts'],data['blockSizes'])}
        for type_,sites in site_info.items():
            starts,sizes = sites
            if len(starts) != len(sizes):
                raise Exception("Start and end sites number is not same")
            for (start,size) in zip(starts,sizes):
                template_ann_seq.add_ann(type_,1,start,start+size-1)
        template_ann_seq.op_not_ann("intron","gene","exon")
        ann_seq = self._create_seq(data['name'],data['chrom'],data['strand'],tx_start,tx_end)
        for type_ in ann_seq.ANN_TYPES:
            data = template_ann_seq.get_ann(type_)
            ann_seq.add_ann(type_,data)
        self._validate(ann_seq,self._foucus_type)
        return ann_seq

class CodingAnnSeqConverter(AnnSeqConverter):
    def __init__(self,foucus_type=None,extra_types=None):
        foucus_type = foucus_type or ['cds','intron','utr_5','utr_3']
        extra_types = extra_types or []
        super().__init__(foucus_type,extra_types)
    def _get_template_ann_seq(self,length):
        """Get template container"""
        ann_seq = AnnSequence()
        ann_seq.length = length
        ann_seq.strand = 'plus'
        ann_seq.id = 'template'
        extra_type = ['exon','ORF','utr','utr_5_potential','utr_3_potential','gene']
        ann_seq.ANN_TYPES = set(self.ANN_TYPES + extra_type)
        ann_seq.init_space()
        return ann_seq
    def _validate(self,ann_seq,focus_types=None):      
        if not is_one_hot(ann_seq,focus_types):
            raise NotOneHotException(ann_seq.id)
        if 'exon' in ann_seq.ANN_TYPES:
            added_exon = np.array([0]*ann_seq.length,dtype='float64')
            for type_ in ['utr_5','utr_3','cds']:
                added_exon += ann_seq.get_ann(type_)   
            if not np.all(added_exon==ann_seq.get_ann('exon')):
                raise Exception("Exon status is not consistent with UTR and CDS")

class UscuSeqConverter(CodingAnnSeqConverter):
    def convert(self,data):
        """Convert zero-based dicitonary data into AnnSequence"""
        tx_start = data['txStart']
        tx_end = data['txEnd']
        #calculate for template container
        cds_start=data['cdsStart'] - tx_start
        cds_end=data['cdsEnd'] - tx_start
        length = tx_end-tx_start+1
        strand = data['strand']
        template_ann_seq = self._get_template_ann_seq(length)
        template_ann_seq.set_ann('gene',1)
        if data['exonCount'] <= 0:
            raise NotPositiveException("exonCount",data['exonCount'])
        else:
            exonStarts = [start_index - tx_start for start_index in data['exonStarts']]
            exonEnds = [end_index - tx_start for end_index in data['exonEnds']]
        #if exon exist,it will add cds,utr(if exists),intron(if exists) in template container.
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
        ann_seq = self._create_seq(name,chrom_id,strand,tx_start,tx_end)
        for type_ in ann_seq.ANN_TYPES:
            data = template_ann_seq.get_ann(type_)
            ann_seq.set_ann(type_,data)
        self._validate(ann_seq,self._foucus_type)
        return ann_seq

class EnsemblSeqConverter(CodingAnnSeqConverter):
    def convert(self,data):
        """Convert zero-based dicitonary data into AnnSequence"""
        tx_start = data['tx_start']
        tx_end = data['tx_end']
        """Create template"""
        length = tx_end-tx_start+1
        template_ann_seq = self._get_template_ann_seq(length)
        """Annotation"""
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
        ann_seq = self._create_seq(data['protein_id'],data['chrom'],
                                   data['strand'],tx_start,tx_end)
        for type_ in ann_seq.ANN_TYPES:
            data = template_ann_seq.get_ann(type_)
            ann_seq.add_ann(type_,data)
        self._validate(ann_seq,self._foucus_type)
        return ann_seq
