import os, sys
import pandas as pd
import warnings
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_bed, write_bed, get_gff_with_attribute, read_gff
from sequence_annotation.utils.utils  import read_fasta
from sequence_annotation.preprocess.gff2bed import gff_info2bed_info
from sequence_annotation.genome_handler.exception import InvalidStrandType

def create_canonical_bed_with_orf(bed,cDNAs,start_codons,stop_codons):
    returned_list = []
    for item in bed.to_dict('record'):
        item_ = dict(item)
        strand = item['strand']
        if strand not in ['+','-']:
            raise InvalidStrandType(strand)
        cDNA = cDNAs[item['id']]
        start_index = -1
        for start_codon in start_codons:
            index = cDNA.find(start_codon)
            if index != -1:
                if start_index != -1:
                    if strand == '+':
                        start_index = min(start_index,index)
                    else:
                        start_index = max(start_index,index)
                else:
                    start_index = index
        #Start codon exists in cDNA
        if start_index != -1:
            stop_index = -1
            for index in range(start_index+3,len(cDNA),3):
                codon = cDNA[index:index+3]
                if codon in stop_codons:
                    stop_index = index + 2
                    break
            #Stop codon exists in cDNA
            if stop_index != -1:
                rel_start = -1
                rel_stop = -1
                length = 0
                block_size = [int(val) for val in item['block_size'].split(',')]
                block_related_start = [int(val) for val in item['block_related_start'].split(',')]
                if strand == '-':
                    block_size = reversed(block_size)
                    block_related_start = reversed(block_related_start)
                for size, block_start in zip(block_size,block_related_start):
                    start = length
                    end = start + size - 1
                    if start <= start_index <= end:
                        rel_start = start_index - start
                        if strand == '+':
                            rel_start += block_start
                        else:
                            rel_start = block_start + size - 1 - rel_start
                    if start <= stop_index <= end:
                        rel_stop = stop_index - start
                        if strand == '+':
                            rel_stop += block_start
                        else:    
                            rel_stop = block_start + size - 1 - rel_stop
                    length += size

                if rel_start != -1 and rel_stop != -1:
                    start = rel_start
                    stop = rel_stop
                    if start < 0 or stop < 0:
                        raise Exception(len(cDNA),rel_start,rel_stop,start_index,stop_index)
                    start += item['start']
                    stop += item['start']
                    item_['thick_start'] = min(start,stop)
                    item_['thick_end'] = max(start,stop)
                else:
                    raise Exception("Data is wrong {}".format(item,rel_start,rel_stop,start_index,stop_index))
            #Stop codon not exists in cDNA
            else:
                warnings.warn("{} has no stop codon".format(item['id']))
                item_['thick_start'] = item_['start']
                item_['thick_end'] = item_['start'] - 1
        else:
            warnings.warn("{} has no ORF".format(item['id']))
            item_['thick_start'] = item_['start']
            item_['thick_end'] = item_['start'] - 1
        returned_list.append(item_)
    returned_bed = pd.DataFrame.from_dict(returned_list) 
    return returned_bed

def create_canonical_bed_without_orf(gff):
    gff = get_gff_with_attribute(gff)
    genes = gff[gff['feature']=='gene']
    ids = set(genes['id'])
    genes = genes.groupby('id')
    exons = gff[gff['feature'].isin(['exon','alt_acceptor','alt_donor'])]
    exons = exons.groupby('parent')
    bed_info_list = []
    for id_ in ids:
        gene = genes.get_group(id_).to_dict('record')[0]
        exons_ = exons.get_group(id_).to_dict('list')
        orf = {'id':id_,'thick_start':gene['start'],'thick_end':gene['start']-1}
        bed_info_list.append(gff_info2bed_info(gene,exons_,orf))
    bed = pd.DataFrame.from_dict(bed_info_list)
    return bed

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",help='If postfix is bed, then it will be used directly'+
                        '. If postfix is gff or gff3, it will be treated as alternative gff , '+
                        'it will be converted to bed first',required=True)
    parser.add_argument("-o", "--output_canonical_bed_path",required=True)
    parser.add_argument("--orf",help="If it is selected, output data will have ORF based on translation table"+
                        "and fasta data",action='store_true')
    parser.add_argument("-f","--fasta_path")
    parser.add_argument("-t","--translation_table_path",
                        help="If it is not provided, then standard codon will be used")
    parser.add_argument("--valid_start_aa", help="Default start amino acid is M")
    parser.add_argument("--valid_stop_aa", help="Default stop amino acid is *")
    
    args = parser.parse_args()
    postfix = args.input_path.split('.')[-1]
    if postfix in ['gff','gff3']:
        gff = read_gff(args.input_path)
        bed = create_canonical_bed_without_orf(gff)
        write_bed(bed,args.output_canonical_bed_path)
        input_bed_path = args.output_canonical_bed_path
    else:
        input_bed_path = args.input_path
    if args.orf:
        cDNA_fasta_path = '.'.join(input_bed_path.split('.')[:-1])+"_cDNA.fasta"
        os.system("rm {}".format(args.fasta_path+".fai"))
        command = 'bedtools getfasta -split -name -s -fi {} -bed {} -fo {}'
        command = command.format(args.fasta_path,input_bed_path,cDNA_fasta_path)
        print(command)
        os.system(command)
        cDNAs = read_fasta(cDNA_fasta_path)

        if args.translation_table_path is None:
            root = os.path.dirname(__file__)
            table_path = os.path.join(root,'standard_codon.tsv')
        else:
            table_path = args.translation_table_path
        translation_table = pd.read_csv(table_path,sep='\t',header=None,comment='#')
        valid_start_aa = args.valid_start_aa or 'M'
        valid_stop_aa = args.valid_stop_aa or '*'
        start_codons = set(translation_table[translation_table[1] == valid_start_aa][0])
        stop_codons = set(translation_table[translation_table[1] == valid_stop_aa][0])

        bed = read_bed(input_bed_path)
        returned_bed = create_canonical_bed_with_orf(bed,cDNAs,start_codons,stop_codons)
        write_bed(returned_bed,args.output_canonical_bed_path)