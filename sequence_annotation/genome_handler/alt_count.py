from .region_extractor import RegionExtractor
from .ann_seq_processor import simplify_seq

class AltCounter:
    def __init__(self):
        self.extractor = RegionExtractor()

    def count(self,anns,gene_map):
        count = []
        for ann in anns:
            count += self.count_per_seq(ann,gene_map)
        return count

    def count_per_seq(self,ann,gene_map):
        if 'gene' not in gene_map.keys():
            raise Exception("Gene must in map's key.")
        simple_seq = simplify_seq(ann,gene_map)
        simple_seq.chromosome_id = ann.chromosome_id or ann.id
        genes = [region for region in self.extractor.extract(simple_seq) if region.ann_type=='gene']
        count = []
        for gene in genes:
            counter = 0
            mRNA = gene.copy()
            mRNA.ann_type = 'mRNA'
            mRNA.id = gene.id+"_mRNA"
            mRNA.parent = gene.id
            subseq = ann.get_subseq(mRNA.start,mRNA.end)
            subseq.id = mRNA.id
            subseq.chromosome_id = mRNA.chromosome_id
            subseq.source = mRNA.source
            regions = self.extractor.extract(subseq)
            for region in regions:
                if region.ann_type == 'alt_intron':
                    counter += 1
            count.append(counter)
        return count

def max_alt_count(seqs,gene_map):
    counter = AltCounter()
    count = counter.count(seqs,gene_map)
    return max(count)
