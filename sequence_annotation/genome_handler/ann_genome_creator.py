import numpy as np
from . import AnnSeqContainer
from . import AnnSequence
from . import InvalidStrandType
from . import DictValidator
from . import Creator
from . import validate_return
from . import NotPositiveException
class AnnGenomeCreator(Creator):
    """Purpose:Make sequences of numeric value represent annotated region occupy or not"""
    def __init__(self,genome_information,uscu_data):
        super().__init__()
        self._data_information = genome_information
        self._uscu_data = uscu_data
        self._ANN_TYPES = self._get_ann_types()
    def _get_ann_types(self):
        return ['intergenic_region','utr_3','utr_5','intron','cds']
    def _validate(self):
        pass
    def create(self):
        self._validate()
        self._validate_genome_info()
        self._result = self._get_init_genome()
        self.__annotate_genome()
    @property
    @validate_return("use method create before access the data")
    def result(self):
        """Get data"""
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
                ann_seq.initSpace()
                genome.add(ann_seq)
        return genome
    def _add_seq_to_genome(self, ann_seq, start_index, end_index):
        chrom_id = ann_seq.chromosome_id
        strand = ann_seq.strand
        chrom = self._result.get(chrom_id+"_"+strand)
        for type_ in chrom.ANN_TYPES:
            seq = ann_seq.get_ann(type_)
            if strand=='minus':
                seq = np.flip(seq, 0)
            chrom.add_ann(type_,seq,start_index,end_index)
    def _create_seq(self,seq_info):
        txStart = seq_info['txStart']
        txEnd = seq_info['txEnd']
        length = txEnd-txStart+1
        chrom_id = seq_info['chrom']
        strand_type = seq_info['strand']
        strand = ""
        if strand_type=='+':
            strand = "plus"
        elif strand_type=='-':
            strand = "minus"
        else:
            raise InvalidStrandType(strand_type)
        nt_number = length
        ann_seq = AnnSequence()
        ann_seq.id = seq_info['name']
        ann_seq.length = nt_number
        ann_seq.ANN_TYPES = self._ANN_TYPES
        ann_seq.chromosome_id = chrom_id
        ann_seq.strand = strand
        ann_seq.source = self._data_information['source']
        ann_seq.initSpace()
        return ann_seq
    def __get_template_ann_seq(self,length):
        """Get template container"""
        ann_seq = AnnSequence()
        ann_seq.length = length
        ann_seq.chromosome_id = 'template'
        ann_seq.strand = 'plus'
        ann_seq.source = 'template'
        ann_seq.id = 'template'
        ann_seq.ANN_TYPES = self._ANN_TYPES+['exon','ORF','utr',
                                             'utr_5_potential','utr_3_potential','gene']
        ann_seq.initSpace()
        return ann_seq
    def __annotate_genome(self):
        """Annotate genome"""
        for seq_data in self._uscu_data:
            self.__annotate_seq(seq_data)
    def __annotate_seq(self,seq_data):
        txStart = seq_data['txStart']
        txEnd = seq_data['txEnd']
        #calculate for template container
        relate_cdsStart=seq_data['cdsStart'] - txStart
        relate_cdsEnd=seq_data['cdsEnd'] - txStart
        length = txEnd-txStart+1
        strand = seq_data['strand']
        chrom_length = self._data_information['chromosome'][seq_data['chrom']]
        if strand== '+':
            gene_start_index = txStart
            gene_end_index = txEnd
        elif strand == '-':
            gene_start_index = chrom_length-1-txEnd
            gene_end_index = chrom_length-1-txStart
        else:
            raise InvalidStrandType(strand)
        template_ann_seq = self.__get_template_ann_seq(length)
        template_ann_seq.set_ann('gene',1)
        if seq_data['exonCount'] <=0:
            raise NotPositiveException("exonCount",seq_data['exonCount'])
        else:
            relate_exonStarts = [start_index - txStart for start_index in seq_data['exonStarts']]
            relate_exonEnds = [end_index - txStart for end_index in seq_data['exonEnds']]
        #if exon exist,it will add cds,utr(if exists),intron(if exists) in template container.
        for exon_index in range(seq_data['exonCount']):
            start = relate_exonStarts[exon_index]
            end = relate_exonEnds[exon_index]
            template_ann_seq.set_ann('exon', 1, start, end)
        if relate_cdsStart <= relate_cdsEnd:
            template_ann_seq.set_ann('ORF', 1 ,relate_cdsStart,relate_cdsEnd)
        template_ann_seq.op_and_ann('cds','exon','ORF')    
        template_ann_seq.op_not_ann('utr','exon','ORF')
        template_ann_seq.op_not_ann('intron','gene','exon')
        utr = template_ann_seq.get_ann('utr')
        if np.any(utr==1):
            if np.any(template_ann_seq.get_ann('cds')==1):
                if  strand== '+':
                    if relate_cdsStart-1>=0:
                        template_ann_seq.set_ann('utr_5_potential',1,0,relate_cdsStart-1)
                    if relate_cdsEnd+1<=length-1:
                        template_ann_seq.set_ann('utr_3_potential',1,relate_cdsEnd+1,length-1)
                elif strand == '-':
                    if relate_cdsStart-1>=0:
                        template_ann_seq.set_ann('utr_3_potential',1,0,relate_cdsStart-1)
                    if relate_cdsEnd+1<=length-1:
                        template_ann_seq.set_ann('utr_5_potential',1,relate_cdsEnd+1,length-1)
                template_ann_seq.op_and_ann('utr_5','utr_5_potential','utr')
                template_ann_seq.op_and_ann('utr_3','utr_3_potential','utr')
            else:
                template_ann_seq.set_ann('utr_5',utr)
        #add template annotated chromosmome to whole annotated chromosmome
        ann_seq =self._create_seq(seq_data)
        total = 0
        for type_ in ann_seq.ANN_TYPES:
            data = template_ann_seq.get_ann(type_)
            ann_seq.set_ann(type_,data)
            total += sum(data)
        if total==ann_seq.length:
            self._add_seq_to_genome(ann_seq,gene_start_index,gene_end_index)
        else:
            raise Exception("Sequence annotation is not filled completely: "+str(total)+"/"+str(ann_seq.length))
