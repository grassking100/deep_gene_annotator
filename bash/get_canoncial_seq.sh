#!/bin/bash
usage(){
 echo "Usage: The pipeline creating canonical gene, cDNA, and peptide"
 echo "  Arguments:"
 echo "    -i  <string>  Path of transcript in gff3 format"
 echo "    -f  <string>  Path of genome in fasta format"
 echo "    -o  <string>  Output root"
 echo "    -t  <string>  Directory of TransDecoder"
 echo "  Options:"
 echo "    -h            Print help message and exit"
 echo "Example: bash get_canoncial_seq.sh -i /home/io/example.gff3 -o canoncial_result -t path_to_transdecoder"
 echo ""
}

while getopts i:f:o:t:h option
 do
  case "${option}"
  in
   i )transcript_gff_path=$OPTARG;;
   f )fasta_path=$OPTARG;;
   o )output_root=$OPTARG;;
   t )transdecoder_root=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$transcript_gff_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi

if [ ! "$fasta_path" ]; then
    echo "Missing option -f"
    usage
    exit 1
fi


if [ ! "$output_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$transdecoder_root" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi


bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
visual_main_root=$script_root/sequence_annotation/visual
transdecoder_result_root=$output_root/transdecoder
canonical_bed_path=$output_root/canonical.bed
canonical_gff_path=$output_root/canonical.gff3
alt_region_gff_path=$output_root/alt_region.gff3
alt_region_id_table_path=$output_root/alt_region_id_table.tsv
canoncial_cDNA_path=$output_root/canoncial_cDNA.fasta

mkdir -p $output_root

#python3 $preprocess_main_root/create_canonical_gene.py -i $transcript_gff_path  \
#-s $alt_region_gff_path -b $canonical_bed_path -g $canonical_gff_path -o $alt_region_id_table_path
bedtools getfasta -name -fi  $fasta_path -bed $canonical_bed_path -split -s -fo $canoncial_cDNA_path

#rm -r $transdecoder_result_root
mkdir -p $transdecoder_result_root
cd $transdecoder_result_root
perl $transdecoder_root/TransDecoder.LongOrfs -S -t $canoncial_cDNA_path > $transdecoder_result_root/long_orf_record.log
perl $transdecoder_root/TransDecoder.Predict --single_best_only -t $canoncial_cDNA_path > $transdecoder_result_root/predict_record.log
