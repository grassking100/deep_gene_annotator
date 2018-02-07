from . import SeqInfoContainer
from . import SeqInformation
from . import validate_return
class RegionExtractor:
    """#Get annotated region information"""
    def __init__(self,ann_seq,frontground_types,background_type):
        self._data = ann_seq.get_one_hot(frontground_types,background_type)
        self._seq_info_genome = None
        self._region_id = None
    def extract(self):
        self._region_id = 0
        self._seq_info_genome = SeqInfoContainer()
        self._parse_regions(self._data)
    @property
    @validate_return("use extract before access the data")
    def result(self):
        return self._seq_info_genome
    def _parse_regions(self,seq):
        for type_ in seq.ANN_TYPES:
            self._parse_of_region(type_)
    def _create_seq_info(self, ann_type, start, end):
        target = SeqInformation()
        region_id_prefix = 'region_'+self._data.chromosome_id+"_"+self._data.strand+"_"
        target.ann_type = ann_type
        target.chromosome_id = self._data.chromosome_id
        target.strand = self._data.strand
        target.source = self._data.source
        target.ann_status = "whole"
        target.id = region_id_prefix + str(self._region_id)
        target.start = start
        target.end = end
        self._region_id += 1
        return target
    def _parse_of_region(self,ann_type):
        one_type_seq = self._data.get_ann(ann_type)
        start = None
        end = None
        for index, sub_seq in enumerate(one_type_seq):
            if start is None:
                if sub_seq==1:
                    start = index
            if start is not None:
                if sub_seq==0:
                    end = index - 1
            if start is not None and end is not None:
                target = self._create_seq_info(ann_type, start, end)
                self._seq_info_genome.add(target)
                start = None
                end = None
        #handle special case
        if start is not None and end is None:
            target = self._create_seq_info(ann_type, start, self._data.length - 1)
            self._seq_info_genome.add(target)
            