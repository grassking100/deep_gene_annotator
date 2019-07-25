from .region_extractor import RegionExtractor

def _create_seq(seq,ann_types):
    new_seq = seq.copy()
    new_seq.clean_space()
    new_seq.ANN_TYPES = ann_types
    new_seq.init_space()
    return new_seq

class ExonHandler:
    def __init__(self):
        self._exon_subtypes = ['utr_5','utr_3','cds']
        self._exon_types = ['internal_exon','external_exon']
        self._internal_subtypes = []
        self._external_subtypes = []
        self._position_names = ['internal','external']
        for ann_type in self._exon_subtypes:
            self._internal_subtypes += ['internal_'+ann_type]
            self._external_subtypes += ['external_'+ann_type]
        self._all_exon_types = self._internal_subtypes + self._external_subtypes + self._exon_types + ['exon'] + self._exon_subtypes

    @property
    def internal_subtypes(self):
        return self._internal_subtypes

    @property
    def external_subtypes(self):
        return self._external_subtypes

    def _get_other_types(self,seq):
        """Get types which are not related with exon"""
        other_types = []
        for type_ in seq.ANN_TYPES:
            if type_ not in self._all_exon_types:
                other_types.append(type_)
        return other_types

    def further_division(self,seq):
        focus_types = []
        division_types = []
        for subtype in self._exon_subtypes + ['exon']:
            if subtype in seq.ANN_TYPES:
                for position in self._position_names:
                    division_types.append('{}_{}'.format(position,subtype))
                focus_types.append(subtype)
        if not focus_types:
            raise Exception("Require sequence has no exon related type.")

        extractor = RegionExtractor()
        exons = extractor.extract(seq,focus_types)
        if not exons:
            return seq
        else:
            other_types = self._get_other_types(seq)
            sorted_exons = sorted(exons, key=lambda x: x.start, reverse=False)
            internal_exons = sorted_exons[1:-1]
            external_exons = [sorted_exons[0],sorted_exons[-1]]
            exon_seq = _create_seq(seq,self._all_exon_types)
            for exon in external_exons:
                exon_seq.add_ann('external_{}'.format(exon.ann_type),1,exon.start,exon.end)
            for exon in internal_exons:
                exon_seq.add_ann('internal_{}'.format(exon.ann_type),1,exon.start,exon.end)
            new_seq = _create_seq(seq,other_types + division_types)
            for type_ in division_types:
                new_seq.add_ann(type_,exon_seq.get_ann(type_))
            for type_ in other_types:
                new_seq.add_ann(type_,seq.get_ann(type_))
            new_seq.processed_status = 'further_division'
            return new_seq

    def discard_external(self,seq):
        ann_types = seq.ANN_TYPES
        for type_ in self._exon_subtypes + ['exon']:
            ann_type = 'external_{}'.format(type_)
            if ann_type in seq.ANN_TYPES:
                ann_types.remove(ann_type)
        new_seq = _create_seq(seq,ann_types)
        for type_ in ann_types:
            new_seq.add_ann(type_,seq.get_ann(type_))
        return new_seq

    def simplify_exon_name(self,seq):
        simplify_exon_types = set()
        for type_ in self._exon_subtypes + ['exon']:
            for position in self._position_names:
                ann_type = '{}_{}'.format(position,type_)
                if ann_type in seq.ANN_TYPES:
                    simplify_exon_types.add(type_)
        simplify_exon_types = list(simplify_exon_types)
        if not simplify_exon_types:
            raise Exception("Require sequence has no exon related type, it gets {}.".format(seq.ANN_TYPES))

        other_types = self._get_other_types(seq)
        new_seq = _create_seq(seq,other_types + simplify_exon_types)
        for place_type in self._position_names:
            for ann_type in simplify_exon_types:
                type_ = "{}_{}".format(place_type,ann_type)
                if type_ in seq.ANN_TYPES:
                    new_seq.add_ann(ann_type,seq.get_ann(type_))
        for type_ in other_types:
            if type_ in seq.ANN_TYPES:
                new_seq.add_ann(type_,seq.get_ann(type_))
        return new_seq
