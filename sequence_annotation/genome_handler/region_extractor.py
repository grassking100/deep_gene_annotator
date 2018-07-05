from . import SeqInfoContainer
from . import SeqInformation
class RegionExtractor:
    """#Get annotated region information"""
    def extract(self,ann_seq,focus_types=None):
        focus_types = focus_types or ann_seq.ANN_TYPES
        if ann_seq.processed_status=="one_hot":
            self._region_id = 0
            seq_info_genome = self._parse_regions(ann_seq,focus_types)
            return seq_info_genome
        else:
            err = "Input sequence's type must be one-hot sequence,not "+(ann_seq.processed_status or "None")
            raise Exception(err)
    def _parse_regions(self,seq,focus_types):
        seq_info_genome = SeqInfoContainer()
        for type_ in focus_types:
            self._parse_of_region(seq_info_genome,seq,type_)
        return seq_info_genome
    def _create_seq_info(self, seq, ann_type, start, end):
        target = SeqInformation()
        region_id_prefix = 'region_'+str(seq.chromosome_id)+"_"+seq.strand+"_"
        target.chromosome_id = seq.chromosome_id
        target.strand = seq.strand
        target.source = seq.source
        target.ann_type = ann_type
        target.ann_status = "whole"
        target.id = region_id_prefix + str(self._region_id)
        target.start = start
        target.end = end
        self._region_id += 1
        return target
    def _parse_of_region(self,seq_info_genome,seq,ann_type):
        one_type_seq = seq.get_ann(ann_type)
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
                target = self._create_seq_info(seq,ann_type,start,end)
                seq_info_genome.add(target)
                start = None
                end = None
        #handle special case
        if start is not None and end is None:
            target = self._create_seq_info(seq, ann_type, start, seq.length - 1)
            seq_info_genome.add(target)