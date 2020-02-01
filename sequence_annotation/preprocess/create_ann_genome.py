import os, sys
import deepdish as dd
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import get_gff_with_attribute, read_gff, read_bed,BASIC_GENE_ANN_TYPES
from sequence_annotation.genome_handler.exception import NotOneHotException
from sequence_annotation.genome_handler.ann_seq_processor import is_one_hot
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.ann_genome_processor import get_backgrounded_genome
from sequence_annotation.preprocess.utils import RNA_TYPES,get_gff_with_intron

STRAND_CONVERT = {'+':'plus','-':'minus'}

class Gff2AnnSeqs:
    def __init__(self,ann_types,one_hot_ann_types):
        self._ANN_TYPES = ann_types
        self._one_hot_ann_types = one_hot_ann_types
    
    @property
    def ANN_TYPES(self):
        return self._ANN_TYPES

    def _create_seq(self,chrom_id,strand,length):
        """Create annotation sequence for returning"""
        ann_seq = AnnSequence()
        ann_seq.id = chrom_id
        ann_seq.length = length
        ann_seq.strand = STRAND_CONVERT[strand]
        ann_seq.ANN_TYPES = self.ANN_TYPES
        ann_seq.chromosome_id = chrom_id
        ann_seq.init_space()
        return ann_seq

    def _validate(self,ann_seq):
        if not is_one_hot(ann_seq,self._one_hot_ann_types):
            raise NotOneHotException(ann_seq.id)

    def convert(self,gff,region_bed,source):
        gff = get_gff_with_attribute(gff)
        is_transcript = gff['feature'].isin(RNA_TYPES)
        transcripts = gff[is_transcript].to_dict('record')
        blocks = gff[~is_transcript].groupby('parent')
        genome = AnnSeqContainer()
        genome.ANN_TYPES = self.ANN_TYPES
        for region in region_bed.to_dict('record'):
            chrom = self._create_seq(region['id'],region['strand'],region['end']-region['start']+1)
            chrom.source = source
            genome.add(chrom)

        for transcript in transcripts:
            blocks_ = blocks.get_group(transcript['id']).to_dict('record')
            for block in blocks_:
                if block['feature'] in self.ANN_TYPES:
                    chrom = genome.get(transcript['chr'])
                    chrom.set_ann(block['feature'],1,block['start']-1,block['end']-1)
                    
        frontgrounded_type = list(self._ANN_TYPES)
        frontgrounded_type.remove('other')
        backgrounded = get_backgrounded_genome(genome,'other',frontgrounded_type)
        for chrom in backgrounded:
            self._validate(chrom)
        return backgrounded

def create_ann_genome(gff,region_bed,souce_name,ann_types,one_hot_ann_types):
    converter = Gff2AnnSeqs(ann_types,one_hot_ann_types)
    genome = converter.convert(gff,region_bed,souce_name)   
    return genome

def main(gff_path,region_bed_path,souce_name,output_path,
         with_alt_region,with_alt_site_region,discard_alt_region,discard_UTR_CDS):
    gff = read_gff(gff_path)
    ann_types = list(BASIC_GENE_ANN_TYPES)
    one_hot_ann_types = list(BASIC_GENE_ANN_TYPES)
    if with_alt_region:
        ann_types += ['exon_skipping','intron_retetion']

    if with_alt_site_region:
        ann_types += ['alt_acceptor','alt_donor']
        one_hot_ann_types += ['alt_acceptor','alt_donor']
    
    if discard_alt_region:
        gff = gff[~gff['feature'].isin(['exon_skipping','intron_retetion'])]
        
    if discard_UTR_CDS:
        gff = gff[~gff['feature'].isin(['UTR','CDS'])]
    
    part_gff = gff[~gff['feature'].isin(ann_types+['UTR', 'mRNA', 'gene'])]
    if len(part_gff) > 0:
        invalid_options = set(part_gff['feature'])
        raise Exception("Invalid features, {}, in GFF, valid options are {}".format(invalid_options,ann_types))
            
    region_bed = read_bed(region_bed_path)
    genome = create_ann_genome(gff,region_bed,souce_name,ann_types,one_hot_ann_types)
    try:
        dd.io.save(output_path,genome.to_dict())
    except OverflowError:
        import pickle
        with open(output_path, 'wb') as fp:
            pickle.dump(genome.to_dict(), fp)
            
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--gff_path",required=True)
    parser.add_argument("-r", "--region_bed_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("-s", "--souce_name",required=True)
    parser.add_argument("--with_alt_region",action='store_true')
    parser.add_argument("--with_alt_site_region",action='store_true')
    parser.add_argument("--discard_alt_region",help='Discard alternative region in GFF',action='store_true')
    parser.add_argument("--discard_UTR_CDS",help='Discard UTR and CDS in GFF',action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
