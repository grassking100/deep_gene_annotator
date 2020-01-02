from ..preprocess.utils import RNA_TYPES
from ..genome_handler.sequence import AnnSequence
from ..genome_handler.region_extractor import RegionExtractor
from ..genome_handler.seq_container import SeqInfoContainer
from ..genome_handler.ann_seq_processor import get_background
from .boundary_process import get_fixed_intron_boundary,get_exon_boundary,fix_splice_pairs,get_splice_pairs,find_substr

def _create_ann_seq(chrom,length,ann_types):
    ann_seq = AnnSequence(ann_types,length)
    ann_seq.id = ann_seq.chromosome_id = chrom
    ann_seq.strand = 'plus'
    return ann_seq

class GeneAnnProcessor:
    def __init__(self,gene_info_extractor,
                 donor_site_pattern=None,accept_site_pattern=None,
                 length_threshold=None,distance=None,
                 gene_length_threshold=None):
        """
        distance : int
            Valid distance  
        donor_site_pattern : str (default : GT)
            Regular expression of donor site
        accept_site_pattern : str (default : AG)
            Regular expression of accept site
        """
        self.gene_info_extractor = gene_info_extractor
        self.extractor = RegionExtractor()
        self.donor_site_pattern = donor_site_pattern or 'GT'
        self.accept_site_pattern = accept_site_pattern or 'AG'
        self.length_threshold = length_threshold or 0
        self.distance = distance or 0
        self.gene_length_threshold = gene_length_threshold or 0
        self._ann_types = ['exon','intron','other']
        
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['gene_info_gene_info_extractor'] = self.gene_info_extractor.get_config()
        config['donor_site_pattern'] = self.donor_site_pattern
        config['accept_site_pattern'] = self.accept_site_pattern
        config['length_threshold'] = self.length_threshold
        config['gene_length_threshold'] = self.gene_length_threshold
        config['distance'] = self.distance
        config['ann_types'] = self._ann_types
        return config
        
    def _validate_gff(self,gff):
        strand = list(set(gff['strand']))
        if len(strand)!=1 or strand[0] != '+':
            raise Exception("Got strands, {}".format(strand))
        
    def _create_ann_seq(self,chrom,length,gff):
        ann_seq = _create_ann_seq(chrom,length,self._ann_types)
        exon_introns = gff[gff['feature'].isin(['exon','intron'])].to_dict('record')
        for item in exon_introns:
            ann_seq.add_ann(item['feature'],1,item['start']-1,item['end']-1)
        other = get_background(ann_seq,['exon','intron'])
        ann_seq.add_ann('other',other)
        return ann_seq

    def _process_ann(self,info):
        info = info.to_data_frame()
        info = info.assign(length=info['end'] - info['start'] + 1)
        fragments = info[info['length'] < self.length_threshold].sort_values(by='length')
        blocks = info[info['length'] >= self.length_threshold]
        if len(blocks)==0:
            raise Exception()
        index=0
        while True:
            if len(fragments) == 0:
                break
            fragments = fragments.reset_index(drop=True)
            fragment = dict(fragments.loc[index])
            rights = blocks[blocks['start'] == fragment['end']+1].to_dict('record')
            lefts = blocks[blocks['end'] == fragment['start']-1].to_dict('record')
            feature = None
            if len(lefts) > 0:
                left = lefts[0]
                feature = left['ann_type']
            elif len(rights) > 0:
                right = rights[0]
                feature = right['ann_type']
            if feature is None:
                index+=1
            else:
                fragment['ann_type'] = feature
                blocks = blocks.append(fragment,ignore_index=True)
                fragments=fragments.drop(index)
                index=0
        blocks_ = SeqInfoContainer()
        blocks_.from_dict({'data':blocks.to_dict('record'),'note':None})
        return blocks_

    def _process_transcript(self,rna,intron_boundarys,ann_seq):
        rna_boundary = rna['start'],rna['end']
        length = rna['end']-rna['start']+1
        if length >= self.gene_length_threshold:
            rna_intron_boundarys = []
            for intron_boundary in intron_boundarys:
                if rna['start'] <= intron_boundary[0] <= intron_boundary[1] <= rna['end']:
                    rna_intron_boundarys.append(intron_boundary)
            rna_intron_boundarys = get_fixed_intron_boundary(rna_boundary,rna_intron_boundarys)
            rna_exon_boundarys = get_exon_boundary(rna_boundary,rna_intron_boundarys)
            for boundary in rna_exon_boundarys:
                ann_seq.add_ann('exon',1,boundary[0]-1,boundary[1]-1)
            for boundary in rna_intron_boundarys:
                ann_seq.add_ann('intron',1,boundary[0]-1,boundary[1]-1)

    def process(self,chrom,length,seq,gff):
        """Get fixed AnnSequence
        Parameters:
        ----------
        chrom : str
            Chromosome id to be chosen
        seq : str
            DNA sequence which its direction is 5' to 3'
        length : int
            Length of chromosome
        gff : pd.DataFrame    
            GFF data about exon and intron
        Returns:
        ----------
        SeqInfoContainer
        """
        self._validate_gff(gff)
        gff = gff[gff['chr']==chrom]
        ann_seq = self._create_ann_seq(chrom,length,gff)
        info = self.extractor.extract(ann_seq)
        gff = self._process_ann(info).to_gff()
        ann_seq = self._create_ann_seq(chrom,length,gff)
        gff = self.gene_info_extractor.extract_per_seq(ann_seq).to_gff()
        splice_pairs = get_splice_pairs(gff)
        ann_donor_sites = [site + 1 for site in find_substr(self.donor_site_pattern,seq)]
        ann_accept_sites = [site + 1 for site in find_substr(self.accept_site_pattern,seq,False)]
        intron_boundarys = fix_splice_pairs(splice_pairs,ann_donor_sites,ann_accept_sites,self.distance)
        fixed_ann_seq = _create_ann_seq(chrom,length,self._ann_types)
        transcripts = gff[gff['feature'].isin(RNA_TYPES)].to_dict('record')
        for transcript in transcripts:
            self._process_transcript(transcript,intron_boundarys,fixed_ann_seq)
        other = get_background(fixed_ann_seq,['exon','intron'])
        fixed_ann_seq.add_ann('other',other)
        info = self.gene_info_extractor.extract_per_seq(fixed_ann_seq)
        gff = info.to_gff()
        return gff
