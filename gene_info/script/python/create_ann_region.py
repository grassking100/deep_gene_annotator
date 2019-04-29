import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../../..")
from argparse import ArgumentParser

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will used mRNA_bed12 data to create annotated genome\n"+
                            "and it will selecte region from annotated genome according to the\n"+
                            "selected_region and selected region will save to output_path in h5 format")
    parser.add_argument("-m", "--mRNA_bed12_path",
                        help="Path of selected mRNA bed file",required=True)
    parser.add_argument("-r", "--selected_region_path",
                        help="Path of selected region bed file",required=True)
    parser.add_argument("-f", "--fai_path",
                        help="Path of fai file",required=True)
    parser.add_argument("-s", "--source_name",
                        help="Genome source name",required=True)
    parser.add_argument("-o", "--output_path",
                        help="Path of output file",required=True)
    parser.add_argument("--saved_root",
                        help="Path of saved_root file",required=True)
    args = vars(parser.parse_args())
    mRNA_bed12_path = args['mRNA_bed12_path']
    selected_region_path = args['selected_region_path']
    fai_path = args['fai_path']
    source_name = args['source_name']
    output_path = args['output_path']
    saved_root = args['saved_root']
    
    #Load library
    from sequence_annotation.genome_handler.sequence import AnnSequence,SeqInformation
    from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser
    from sequence_annotation.genome_handler.ann_seq_converter import CodingBedSeqConverter
    from sequence_annotation.genome_handler.seq_container import AnnSeqContainer,SeqInfoContainer
    from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
    from sequence_annotation.genome_handler.ann_seq_extractor import AnnSeqExtractor
    from sequence_annotation.genome_handler.ann_genome_processor import get_backgrounded_genome
    import pandas as pd
    import deepdish as dd
    #Read bed files of selected mRNA and selected region
    data = pd.read_csv(mRNA_bed12_path,header=None,sep='\t').to_dict('record')
    araound_data = pd.read_csv(selected_region_path,header=None,sep='\t').to_dict('record')
    #Read chromosome length file
    chrom_info = pd.read_csv(fai_path,header=None,sep='\t')
    #chrom_id = [chr_.replace('Chr','') for chr_ in chrom_info[0]]
    chrom_id = chrom_info[0]
    chrom_length = chrom_info[1]
    chrom_info = {}
    for id_,length in zip(chrom_id,chrom_length):
        chrom_info[str(id_)] = length
    #Parse the bed file and convert its data to AnnSeqContainer
    parser_12 = BedInfoParser()
    parsed = parser_12.parse(data)
    converter = CodingBedSeqConverter()
    ann_seqs = AnnSeqContainer()
    ann_seqs.ANN_TYPES = converter.ANN_TYPES
    print("AnnSeq number:"+str(len(parsed)))
    for index,item in enumerate(parsed):
        seq = converter.convert(item)
        ann_seqs.add(seq)
    chroms = set()
    for seq in ann_seqs:
        chroms.add(seq.chromosome_id+"_"+seq.strand)
    genome_creator = AnnGenomeCreator()
    genome = genome_creator.create(ann_seqs,{'chromosome':chrom_info,'source':source_name})
    backgrounded_genome = get_backgrounded_genome(genome,'other')
    #Parse the bed file and convert its data to SeqInfoContainer
    parser_6 = BedInfoParser(bed_type='bed_6')
    araound_data = parser_6.parse(araound_data)
    seq_infos = SeqInfoContainer()
    for item in araound_data:
        seq_info = SeqInformation()
        seq_info.chromosome_id = item['chrom']
        seq_info.start = item['chromStart']
        seq_info.end = item['chromEnd']
        seq_info.strand = item['strand']
        seq_info.id = item['name']
        seq_infos.add(seq_info)
    parsed_num = len(seq_infos)
    print("Parsed number:"+str(parsed_num))
    #Extract selected AnnSeqContainer
    extractor = AnnSeqExtractor()
    extracted = extractor.extract(backgrounded_genome,seq_infos)
    region_num = len(extracted)
    print("Region number:"+str(region_num))
    dd.io.save(output_path,extracted.to_dict())
    with open(saved_root+'/recurrent_count.stats','w') as fp:
        fp.write("Parsed number:"+str(parsed_num)+"\n")
        fp.write("Region number:"+str(region_num)+"\n")
