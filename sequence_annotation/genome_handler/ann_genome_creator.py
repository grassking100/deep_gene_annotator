import numpy as np
from . import AnnSeqContainer
from . import AnnSequence
from . import ReturnNoneException
from . import InvalidStrandType
from . import DictValidator
from . import Creator
from . import validate_return
from . import NotPositiveException
import pdb
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
        validator = DictValidator(self._data_information)
        validator.key_must_included = ['source','chromosome']
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
                                             'utr_5_potential','utr_3_potential']
        ann_seq.initSpace()
        return ann_seq
    def __annotate_genome(self):
        """Annotate genome"""
        for row in self._uscu_data:
            txStart = row['txStart']
            txEnd = row['txEnd']
            #calculate for template container
            relate_cdsStart=row['cdsStart'] - txStart
            relate_cdsEnd=row['cdsEnd'] - txStart
            length = txEnd-txStart+1
            strand = row['strand']
            template_ann_seq = self.__get_template_ann_seq(length)
            gene_start_index = txStart
            gene_end_index = txEnd
            if row['exonCount'] <=0:
                raise NotPositiveException("exonCount",row['exonCount'])
            else:
                relate_exonStarts = [start_index - txStart for start_index in row['exonStarts']]
                relate_exonEnds = [end_index - txStart for end_index in row['exonEnds']]
                #if exon exist,it will add cds,utr(if exists),intron(if exists) in template container.
                for exon_index in range(row['exonCount']):
                    start = relate_exonStarts[exon_index]
                    end = relate_exonEnds[exon_index]
                    template_ann_seq.set_ann('exon', 1, start, end)
                if relate_cdsStart <= relate_cdsEnd:
                    template_ann_seq.set_ann('ORF', 1 ,relate_cdsStart,relate_cdsEnd)
                template_ann_seq.op_and_ann('cds','exon','ORF')    
                template_ann_seq.op_not_ann('utr','exon','ORF')
                template_ann_seq.op_not_ann('intron','ORF','exon')
                utr = template_ann_seq.get_ann('utr')
                if np.any(utr==1):
                    chrom_length = self._data_information['chromosome'][row['chrom']]
                    utr_false_index=np.where(utr==0)[0]
                    if len(utr_false_index) > 0:
                        utr_split_index_start=utr_false_index[0]
                        utr_split_index_end=utr_false_index[-1]
                        if  strand== '+':
                            if utr_split_index_start-1 >= 0:
                                template_ann_seq.set_ann('utr_5_potential',1,0,utr_split_index_start-1)
                            if utr_split_index_end +1 < length:
                                template_ann_seq.set_ann('utr_3_potential',1,utr_split_index_end+1,length-1)
                        elif strand == '-':
                            if utr_split_index_start-1 >= 0:
                                template_ann_seq.set_ann('utr_3_potential',1,0,utr_split_index_start-1)
                            if utr_split_index_end +1 < length:
                                template_ann_seq.set_ann('utr_5_potential',1,utr_split_index_end+1,length-1)
                            gene_start_index = chrom_length-1-txEnd
                            gene_end_index = chrom_length-1-txStart
                        else:
                            raise InvalidStrandType(strand)
                        template_ann_seq.op_and_ann('utr_5','utr_5_potential','utr')
                        template_ann_seq.op_and_ann('utr_3','utr_3_potential','utr')
                    else:
                        template_ann_seq.set_ann('utr_5',utr)
            #add template annotated chromosmome to whole annotated chromosmome
            ann_seq =self._create_seq(row)
            for type_ in ann_seq.ANN_TYPES:
                ann_seq.set_ann(type_,template_ann_seq.get_ann(type_))
            self._add_seq_to_genome(ann_seq,gene_start_index,gene_end_index)
