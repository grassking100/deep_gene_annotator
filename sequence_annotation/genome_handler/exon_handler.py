from .sequence import AnnSequence
from .region_extractor import RegionExtractor
class ExonHandler:
    @property
    def naive_exon_type(self):
        return self._naive_exon_types
    @property
    def internal_exon_types(self):
        return self._internal_exon_types
    @property
    def external_exon_types(self):
        return self._external_exon_types
    def __init__(self):
        self._naive_exon_types = ['utr_5','utr_3','cds','exon']
        self._position_names = ['internal','external'] 
        self._internal_exon_types = []
        self._external_exon_types = []
        for ann_type in self._naive_exon_types:
            self._internal_exon_types += ['internal_'+ann_type]
        for ann_type in self._naive_exon_types:
            self._external_exon_types += ['external_'+ann_type]
    def further_division(self,seq):
        if 'exon' not in seq.ANN_TYPES:
            raise Exception("Require sequence has exon type.")
        all_exon_types = self._external_exon_types + self._internal_exon_types+ self._naive_exon_types
        other_types = self._get_other_types(seq)
        extractor = RegionExtractor()
        exons = extractor.extract(seq,['exon'])
        if len(exons)==0:
            return seq
        else:
            sorted_exons = sorted(exons, key=lambda x: x.start, reverse=False)
            internal_exons = sorted_exons[1:-1]
            external_exons = [sorted_exons[0],sorted_exons[-1]]
            exon_seq = self._create_seq(seq,all_exon_types)
            for type_ in self._naive_exon_types:
                exon_seq.add_ann(type_,seq.get_ann(type_))
            for exon in external_exons:
                exon_seq.add_ann('external_exon',1,exon.start,exon.end)
            for exon in internal_exons:
                exon_seq.add_ann('internal_exon',1,exon.start,exon.end)
            for place_type in self._position_names:
                for ann_type in ['utr_5','utr_3','cds']:
                    exon_seq.op_and_ann(place_type+"_"+ann_type,place_type+'_exon',ann_type)
            other_types = self._get_other_types(seq)
            new_seq = self._create_seq(seq,other_types + self._internal_exon_types + self._external_exon_types)
            for type_ in self._internal_exon_types + self._external_exon_types:
                new_seq.add_ann(type_,exon_seq.get_ann(type_))
            for type_ in other_types:
                new_seq.add_ann(type_,seq.get_ann(type_))
            new_seq.processed_status = 'further_division'
            return new_seq
    def discard_external(self,seq):
        other_types = self._get_other_types(seq)
        new_seq = self._create_seq(seq,other_types + self._internal_exon_types)
        for type_ in  [new_seq.ANN_TYPES + other_types]:
            if type_ in seq.ANN_TYPES:
                new_seq.add_ann(type_,seq.get_ann(type_))
        return new_seq
    def _create_seq(self,seq,ann_types):
        #processed_status = seq.processed_status
        new_seq = AnnSequence().from_dict(seq.to_dict(without_data=False))
        new_seq.clean_space()
        new_seq.ANN_TYPES = ann_types
        new_seq.init_space()
        #new_seq.processed_status = processed_status
        return new_seq
    def _get_other_types(self,seq):
        """Get types which are not related with exon"""
        other_types = list(seq.ANN_TYPES)
        for type_ in self._naive_exon_types + self._internal_exon_types + self._external_exon_types:
            if type_ in other_types:
                other_types.remove(type_)
        return other_types
    def simplify_exon_name(self,seq):
        other_types = self._get_other_types(seq)
        new_seq = self._create_seq(seq,other_types + self._naive_exon_types)
        for place_type in self._position_names:
            for ann_type in self._naive_exon_types:
                type_ = place_type+"_"+ann_type
                if type_ in seq.ANN_TYPES:
                    new_seq.add_ann(ann_type,seq.get_ann(type_))
        for type_ in other_types:
            if type_ in seq.ANN_TYPES:
                new_seq.add_ann(type_,seq.get_ann(type_))
        return new_seq