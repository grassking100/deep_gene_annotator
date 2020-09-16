import sys,os
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import CONSTANT_LIST


GENE_TYPES = CONSTANT_LIST(['gene','transposable_element','transposable_element_gene','pseudogene'])
PRIMARY_MIRNA_TPYES = CONSTANT_LIST(['miRNA_primary_transcript'])

TRANSCRIPT_TYPES = CONSTANT_LIST([
    'mRNA','pseudogenic_tRNA','pseudogenic_transcript','antisense_lncRNA',
    'lnc_RNA','antisense_RNA','transcript_region','transposon_fragment',
    'tRNA','snRNA','ncRNA','snoRNA','rRNA','transcript','pre_miRNA']+PRIMARY_MIRNA_TPYES)
EXON_TYPES = CONSTANT_LIST(['exon','pseudogenic_exon'])
CDS_TYPES = CONSTANT_LIST(['CDS'])
FIVE_PRIME_UTR_TYPES = CONSTANT_LIST(['five_prime_UTR'])
THREE_PRIME_UTR_TYPES = CONSTANT_LIST(['three_prime_UTR'])
PROTEIN_TYPES = CONSTANT_LIST(['protein'])
UORF_TYPES = CONSTANT_LIST(['uORF'])
MIRNA_TPYES = CONSTANT_LIST(['miRNA'])
ALL_TYPES = GENE_TYPES+TRANSCRIPT_TYPES+EXON_TYPES+CDS_TYPES+FIVE_PRIME_UTR_TYPES+THREE_PRIME_UTR_TYPES+PROTEIN_TYPES+UORF_TYPES+MIRNA_TPYES

#INTRON_TYPES=CONSTANT_LIST(['intron'])
#UTR_TYPES=CONSTANT_LIST([FIVE_PRIME_UTR_TYPE,THREE_PRIME_UTR_TYPE,UTR_TYPE])
#SUBEXON_TYPES=CONSTANT_LIST([FIVE_PRIME_UTR_TYPE,THREE_PRIME_UTR_TYPE,CDS_TYPE,UTR_TYPE])
