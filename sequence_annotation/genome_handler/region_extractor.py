from .seq_container import SeqInfoContainer
from .sequence import SeqInformation
from .ann_seq_processor import is_one_hot,simplify_seq
from .exception import NotOneHotException

class RegionExtractor:
    """#Get annotated region information"""
    def __init__(self):
        self._region_id = 0

    def extract(self,ann_seq,focus_types=None):
        focus_types = focus_types or ann_seq.ANN_TYPES
        if ann_seq.processed_status=="one_hot" or is_one_hot(ann_seq,focus_types):
            seq_infos = self._parse_regions(ann_seq,focus_types)
            return seq_infos
        else:
            raise NotOneHotException(ann_seq.id)

    def _parse_regions(self,seq,focus_types):
        seq_infos = SeqInfoContainer()
        for type_ in focus_types:
            temp = self._parse_of_region(seq,type_)
            seq_infos.add(temp)
        return seq_infos

    def _create_seq_info(self, seq, ann_type, start, end):
        self._region_id += 1
        target = SeqInformation()
        target.note = seq.note
        target.source = seq.source
        target.chromosome_id = seq.chromosome_id
        target.strand = seq.strand
        target.start = start
        target.end = end
        target.ann_type = ann_type
        target.ann_status = 'whole'
        target.parent = seq.id
        target.id = str(seq.id)+"_"+str(ann_type)+"_"+str(self._region_id)
        return target

    def _parse_of_region(self,seq,ann_type):
        self._region_id = 0
        seq_infos = SeqInfoContainer()
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
                seq_infos.add(target)
                start = None
                end = None
        #handle special case
        if start is not None and end is None:
            target = self._create_seq_info(seq, ann_type, start, seq.length - 1)
            seq_infos.add(target)
        return seq_infos

class GeneInfoExtractor:
    def __init__(self):
        self.extractor = RegionExtractor()

    def extract(self,anns,simply_map):
        seq_infos = SeqInfoContainer()
        for ann in anns:
            seq_infos.add(self.extract_per_seq(ann,simply_map))
        return seq_infos

    def extract_per_seq(self,ann,simply_map):
        seq_infos = SeqInfoContainer()
        simple_seq = simplify_seq(ann,simply_map)
        simple_seq.chromosome_id = ann.chromosome_id or ann.id
        genes = [region for region in self.extractor.extract(simple_seq) if region.ann_type=='gene']
        seq_infos.add(genes)
        mRNAs = []
        for gene in genes:
            mRNA = gene.copy()
            mRNA.ann_type = 'mRNA'
            mRNA.id = gene.id+"_mRNA"
            mRNA.parent = gene.id
            mRNAs.append(mRNA)
        seq_infos.add(mRNAs)
        for mRNA in mRNAs:
            subseq = ann.get_subseq(mRNA.start,mRNA.end)
            subseq.id = mRNA.id
            subregions = self.extractor.extract(subseq)
            for region in subregions:
                temp = region.copy()
                temp.start += mRNA.start
                temp.end += mRNA.start
                seq_infos.add(temp)
        return seq_infos