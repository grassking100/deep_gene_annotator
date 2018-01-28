from . import USCU_TableParser
from . import numpy as np
#Purpose:Make sequences of numeric value represent annotated region occupy or not
#Input:GTF file and genome information about chromosomes' id and its length
#Output:A Tree structure made from dictionary,its node stored sequence of numeric value(integer for count,float for frquency)
class GenomeAnnotator:
    def __init__(self,genome_information,gtf_file):
        self.__ANNOTATION_TYPES=['intergenic_region','utr_3','utr_5','intron','cds']
        self.__genome_information=genome_information
        self.__parser=USCU_TableParser(gtf_file) 
        self.__genome=self.__create_genome_list()   
        self.__normalized_genome=self.__create_genome_list()
        self.__annotation()
        self.__normalize()
    def __create_genome_list(self):
        genome={}
        for seq_key in self.__genome_information.keys():
            nt_number=self.__genome_information[seq_key]
            genome[seq_key]={'+':{},'-':{}}
            for annotation_type in self.__ANNOTATION_TYPES:
                value=0
                if annotation_type=="intergenic_region":
                    value=True
                genome[seq_key]['+'][annotation_type]=np.array([value]*nt_number)
                genome[seq_key]['-'][annotation_type]=np.array([value]*nt_number)
        return genome
    def __annotation_per_region(self,region_information):
        region_information=region_information
        seq=self.__genome[region_information['chrom']][region_information['strand']]
        txStart=region_information['txStart']
        seq_len=region_information['txEnd']-txStart+1
        relative_cdsStart=region_information['cdsStart']-txStart
        relative_cdsEnd=region_information['cdsEnd']-txStart
        exon=np.array([False]*seq_len)
        cds_range=np.array([False]*seq_len)
        if region_information['cdsStart']!=region_information['cdsEnd']:
            cds_range[relative_cdsStart:relative_cdsEnd+1]=True
        for i in range(region_information['exonCount']):
            start=region_information['exonStarts'][i]-txStart
            end=region_information['exonEnds'][i]-txStart
            exon[start:end+1]=True
        intron=np.invert(exon)
        utr=np.bitwise_and(exon,np.invert(cds_range))
        cds=np.bitwise_and(exon,cds_range)
        (utr_5,utr_3)=self.__utr_handler(utr,region_information['exonCount'],region_information['strand'],relative_cdsStart,relative_cdsEnd)
        origin_region_index=range(txStart,txStart+seq_len)
        seq['utr_5'][origin_region_index]+=utr_5.astype(int)
        seq['utr_3'][origin_region_index]+=utr_3.astype(int)
        seq['cds'][origin_region_index]+=cds.astype(int)
        seq['intron'][origin_region_index]+=intron.astype(int)
        seq['intergenic_region'][origin_region_index]=np.array([False]*seq_len)
    def __utr_handler(self,utr,exonCount,strand,relative_cdsStart,relative_cdsEnd):
        seq_len=len(utr)
        utr_5_range=np.array([False]*seq_len)
        utr_3_range=np.array([False]*seq_len)
        if (relative_cdsEnd-relative_cdsStart+1)==seq_len:
            return (utr_5_range,utr_3_range)
        if exonCount==0:
            utr_5_range=np.array([True]*seq_len)
            return (utr_5_range,utr_3_range)
        utr_false_index=np.where(utr==False)[0]
        if len(utr_false_index)==0:
            print(utr,exonCount,strand,relative_cdsStart,relative_cdsEnd)
        utr_split_index_start=utr_false_index[0]
        utr_split_index_end=utr_false_index[-1]
        if strand=='+':
            utr_5_range[0:utr_split_index_start]=True
            utr_3_range[utr_split_index_end+1:seq_len]=True
        elif strand=="-":
            utr_3_range[0:utr_split_index_start]=True
            utr_5_range[utr_split_index_end+1:seq_len]=True
        else:
            raise Exception("Unexpected strand presentation:"+str(strand))
        utr_5=np.bitwise_and(utr_5_range,utr)
        utr_3=np.bitwise_and(utr_3_range,utr)
        return (utr_5,utr_3)
    def __annotation(self):
        names=self.__parser.names
        for name in names:
            region_information=self.__parser.get_data(name)
            self.__annotation_per_region(region_information)
        for chrom_id in self.__genome_information.keys():
            for strand,number in zip(['+','-'],[self.__parser.plus_strand_number,self.__parser.minus_strand_number]):         
                sub_seq=self.__genome[chrom_id][strand]
                if number>0:
                    count=number
                else:
                    count=1
                sub_seq['intergenic_region']=sub_seq['intergenic_region'].astype(int)*count
    def __normalize(self):
        sum_of_seq={}
        types=['utr_3','utr_5','intron','cds']
        for chrom_id in self.__genome_information.keys():
            sum_of_seq[chrom_id]={}
            for strand in ['+','-']:
                sum_of_seq[chrom_id][strand]=[0]*self.__genome_information[chrom_id]
                for annotation_type in types:
                    sum_of_seq[chrom_id][strand]+=self.__genome[chrom_id][strand][annotation_type]
        for chrom_id in self.__genome_information.keys():
            for strand in ['+','-']:            
                number=sum_of_seq[chrom_id][strand]
                sub_normalized_seq=self.__normalized_genome[chrom_id][strand]
                sub_seq=self.__genome[chrom_id][strand]
                for annotation_type in types:
                    temp=sub_seq[annotation_type]/number
                    sub_normalized_seq[annotation_type]=np.nan_to_num(temp)
                intergenic_region=sub_seq['intergenic_region']
                sub_normalized_seq['intergenic_region']=intergenic_region.astype(bool).astype(int)
    @staticmethod
    def flatten_data(genome):
        flatten_genome=[]
        for chromosome_id in genome.keys():
            for strand in ['+','-']:
                sub_genome=genome[chromosome_id][strand]
                for annotation_type in sub_genome.keys():
                    seq=genome[chromosome_id][strand][annotation_type]
                    temp={}
                    temp['sequence']=seq
                    temp['chromosome_id']=chromosome_id
                    temp['strand']=strand
                    temp['annotation_type']=annotation_type
                    flatten_genome.append(temp)
        return flatten_genome
    def get_genome(self,is_normalize=True,chromosome_id=None,strand=None,annotation_type=None,sequence_range=None):
        genome=self.__genome
        if is_normalize:
            genome=self.__normalized_genome
        if chromosome_id is None:
            return genome
        elif strand is None:
            return genome[chromosome_id]
        elif annotation_type is None:
            return genome[chromosome_id][strand]
        elif sequence_range is None:
            return genome[chromosome_id][strand][annotation_type]
        else:
            return genome[chromosome_id][strand][annotation_type][sequence_range]