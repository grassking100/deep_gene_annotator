from . import AnnSequence
from . import RegionExtractor
class ExonHandler():
    def __init__(self):
        self._naive_exon_types = ['utr_5','utr_3','cds','exon']
        self._position_names = ['internal','external'] 
        self._internal_exon_types = []
        self._external_exon_types = []
        for ann_type in self._naive_exon_types:
            self._internal_exon_types += ['internal_'+ann_type]
        for ann_type in self._naive_exon_types:
            self._external_exon_types += ['external_'+ann_type]
    def is_valid(self,seq):
        exon_related_types = ['utr_5','utr_3','cds','exon']
        if len(set(exon_related_types) & set(seq.ANN_TYPES))!=len(exon_related_types):
            print(set(exon_related_types) & set(seq.ANN_TYPES))
            raise Exception("Require sequence has utr_5,utr_3,cds,and exon")
        test_seq = self._create_seq(seq,exon_related_types + ['merged']) 
        for type_ in ['utr_5','utr_3','cds']:
            test_seq.add_ann('merged',seq.get_ann(type_))
        return np.all(test_seq.get_ann('merged')==seq.get_ann('exon'))
    def further_division(self,seq):
        if 'exon' not in seq.ANN_TYPES:
            raise Exception("Require sequence has exon type.")
        all_exon_types = self._external_exon_types + self._internal_exon_types+ self._naive_exon_types
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
                exon_seq.set_ann(type_,seq.get_ann(type_))
            for exon in external_exons:
                exon_seq.set_ann('external_exon',1,exon.start,exon.end)
            for exon in internal_exons:
                exon_seq.set_ann('internal_exon',1,exon.start,exon.end)
            for place_type in self._position_names:
                for ann_type in ['utr_5','utr_3','cds']:
                    exon_seq.op_and_ann(place_type+"_"+ann_type,place_type+'_exon',ann_type)
            other_types = self._get_other_types(seq)
            new_seq = self._create_seq(seq,other_types + self._internal_exon_types + self._external_exon_types)
            for type_ in self._internal_exon_types + self._external_exon_types:
                new_seq.set_ann(type_,exon_seq.get_ann(type_))
            for type_ in other_types:
                new_seq.set_ann(type_,seq.get_ann(type_))
            new_seq.processed_status = 'further_division'
            return new_seq
    def discard_external(self,seq):
        other_types = self._get_other_types(seq)
        new_seq = self._create_seq(seq,other_types + self._internal_exon_types)
        for type_ in  new_seq.ANN_TYPES:
            if type_ in seq.ANN_TYPES:
                new_seq.set_ann(type_,seq.get_ann(type_))
        for type_ in other_types:
            if type_ in seq.ANN_TYPES:
                new_seq.set_ann(type_,seq.get_ann(type_))
        return new_seq
    def _create_seq(self,seq,ann_types):
        new_seq = AnnSequence().from_dict(seq.to_dict())
        new_seq.clean_space()
        new_seq.ANN_TYPES = ann_types
        new_seq.init_space()
        return new_seq
    def _get_other_types(self,seq):
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
                    new_seq.set_ann(ann_type,seq.get_ann(type_))
        for type_ in other_types:
            if type_ in seq.ANN_TYPES:
                new_seq.set_ann(type_,seq.get_ann(type_))
        return new_seq