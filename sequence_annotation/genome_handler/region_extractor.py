from abc import abstractmethod
from .seq_container import SeqInfoContainer
from .sequence import SeqInformation
from .ann_seq_processor import is_binary,simplify_seq
from .exception import NotBinaryException

class RegionExtractor:
    """#Get annotated region information"""
    def __init__(self):
        self._region_id = 0

    def extract(self,ann_seq,focus_types=None):
        focus_types = focus_types or ann_seq.ANN_TYPES
        if ann_seq.processed_status=="binary" or is_binary(ann_seq,focus_types):
            seq_infos = self._parse_regions(ann_seq,focus_types)
            return seq_infos
        else:
            raise NotBinaryException(ann_seq.id)

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
        target.id = "{}_{}_{}".format(seq.id,ann_type,self._region_id)
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

class IInfoExtractor:
    @abstractmethod
    def extract(self,anns,simply_map):
        pass
    @abstractmethod
    def extract_per_seq(self,ann,simply_map):
        pass

class GeneInfoExtractor(IInfoExtractor):
    def __init__(self):
        self.extractor = RegionExtractor()
        self.alt = False
        self.alt_num = 0
        self._alt_region_id = 0

    def extract(self,anns,simply_map):
        seq_infos = SeqInfoContainer()
        for ann in anns:
            seq_infos.add(self.extract_per_seq(ann,simply_map))
        return seq_infos

    def extract_per_seq(self,ann,simply_map):
        if 'gene' not in simply_map.keys():
            raise Exception("Gene must in map's key.")
        seq_infos = SeqInfoContainer()
        simple_seq = simplify_seq(ann,simply_map)
        simple_seq.chromosome_id = ann.chromosome_id or ann.id
        genes = [region for region in self.extractor.extract(simple_seq) if region.ann_type=='gene']
        seq_infos.add(genes)
        for gene in genes:
            mRNA = gene.copy()
            mRNA.ann_type = 'mRNA'
            mRNA.id = gene.id+"_mRNA"
            mRNA.parent = gene.id
            subseq = ann.get_subseq(mRNA.start,mRNA.end)
            subseq.id = mRNA.id
            subseq.chromosome_id = mRNA.chromosome_id
            subseq.source = mRNA.source
            regions = self.extractor.extract(subseq)
            if self.alt:
                self._add_alt_regions(mRNA,regions,seq_infos)
            else:
                self._add_regions(mRNA,regions,seq_infos)
        return seq_infos

    def _add_regions(self,mRNA,regions,seq_infos):
        seq_infos.add(mRNA)
        for region in regions:
            copied = region.copy()
            copied.start += mRNA.start
            copied.end += mRNA.start
            seq_infos.add(copied)

    def _add_alt_regions(self,mRNA,regions,seq_infos):
        alt_introns = []
        others = []
        for region in regions:
            if region.ann_type == 'alt_intron':
                alt_introns.append(region)
            else:
                others.append(region)

        if len(alt_introns) > self.alt_num:
            #warn = "There are {} alternative statuses in {}, it will be discarded."
            #warnings.warn(warn.format(len(alt_introns),mRNA.id))
            return

        if not alt_introns:
            self._add_regions(mRNA,regions,seq_infos)
        else:
            alt_statuses = []
            max_val = 2**len(alt_introns)-1
            format_ = "0{}b".format(len(alt_introns))

            for id_ in range(0,max_val+1):
                bits = [int(v) for v in list(format(id_,format_))]
                alt_statuses.append(bits)

            for mRNA_index,alt_ids in enumerate(alt_statuses):
                copied_mRNA = mRNA.copy()
                copied_mRNA.id = "{}_{}".format(mRNA.id,mRNA_index+1)
                seq_infos.add(copied_mRNA)

                alt_seqs = []
                for id_, alt_intron in enumerate(alt_introns):
                    alt = alt_intron.copy()
                    if alt_ids[id_] == 1:
                        alt.ann_type = 'exon'
                    else:
                        alt.ann_type = 'intron'
                    alt_seqs.append(alt)

                for region in others+alt_seqs:
                    self._alt_region_id += 1
                    new_region = copied_mRNA.copy()
                    new_region.ann_type = region.ann_type
                    new_region.start = region.start + copied_mRNA.start
                    new_region.end = region.end + copied_mRNA.start
                    new_region.parent = copied_mRNA.id
                    new_region.id = "{}_{}_{}".format(copied_mRNA.id,new_region.ann_type,self._alt_region_id)
                    seq_infos.add(new_region)
