#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline predict with Augustus"
 echo "  Arguments:"
 echo "    -a  <string>  Root of augustus"
 echo "    -o  <string>  Root of saved result"
 echo "    -t  <string>  Root of transdecoder result"
 echo "    -f  <string>  Testing fasta"
 echo "    -r  <string>  Region table path"
 echo "    -s  <string>  Species name"
 echo "    -d  <int>     Flanking distance around gene of each direction"
 echo "    -m  <string>  The gene model to be used"
 echo "    -a  <bool>    Output prediction by alternatives_from_sampling [default: false]"
 echo "  Options:"
 echo "    -g  <string>  Testing answer in GFF format"
 echo "    -h            Print help message and exit"
 echo "Example: bash predict_with_augustus.sh -o output_root -f test.fasta -g test.gff3 -s arabidopsis_2020_02_28_1 -d 1000 -a -m partial -a /root/augustus -t /root/transdecoder"
 echo ""
}

while getopts a:o:t:f:g:r:s:d:m:ah option
 do
  case "${option}"
  in
   a )augustus_root=$OPTARG;;
   o )output_root=$OPTARG;;
   t )transdecoder_root=$OPTARG;;
   f )test_fasta_path=$OPTARG;;
   g )test_gff_path=$OPTARG;;
   r )region_table_path=$OPTARG;;
   s )species=$OPTARG;;
   d )flanking=$OPTARG;;
   m )genemodel=$OPTARG;;
   a )alternatives_from_sampling=on;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$augustus_root" ]; then
    echo "Missing option -a"
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

if [ ! "$region_table_path" ]; then
    echo "Missing option -r"
    usage
    exit 1
fi

if [ ! "$test_fasta_path" ]; then
    echo "Missing option -f"
    usage
    exit 1
fi

if [ ! "$species" ]; then
    echo "Missing option -s"
    usage
    exit 1
fi

if [ ! "$flanking" ]; then
    echo "Missing option -d"
    usage
    exit 1
fi

if [ ! "$genemodel" ]; then
    echo "Missing option -m"
    usage
    exit 1
fi

if [ ! "$alternatives_from_sampling" ]; then
    alternatives_from_sampling=off
fi


bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
src_root=$bash_root/..
preprocess_main_root=$src_root/sequence_annotation/preprocess
process_main_root=$src_root/sequence_annotation/process
AUGUSTUS_SCRIPTS_ROOT=$augustus_root/scripts
species_path=$AUGUSTUS_CONFIG_PATH/species/$species

export AUGUSTUS_CONFIG_PATH=$augustus_root/config
export augustus_bin_path=$augustus_root/bin
mkdir -p $output_root
cd $output_root

transcript_gff_path=$test_root/test.gff3
canonical_bed_path=$output_root/canonical.bed
canonical_gff_path=$output_root/canonical.gff3
alt_region_gff_path=$output_root/alt_region.gff3
alt_region_id_table_path=$output_root/alt_region_id_table.tsv

augustus --species=$species $test_fasta_path --UTR=on --alternatives-from-sampling=$alternatives_from_sampling --genemodel=$genemodel --gff3=on > $transcript_gff_path

python3 $preprocess_main_root/create_canonical_gene.py -i $transcript_gff_path -s $alt_region_gff_path \
-b $canonical_bed_path -g $canonical_gff_path -o $alt_region_id_table_path

if [ "$test_gff_path" ]; then
    performance_root=$output_root/performance
    mkdir -p $performance_root
    python3 $process_main_root/performance.py -p $canonical_gff_path -a $test_gff_path -r $region_table_path -s $performance_root -t ordinal_id_wo_strand
fi

canoncial_cDNA_path=$output_root/canoncial_cDNA.fasta
bedtools getfasta -fi $test_fasta_path -bed $canonical_bed_path -split -s -fo $canoncial_cDNA_path

transdecoder_result_root=$output_root/transdecoder
mkdir -p $transdecoder_result_root
cd $transdecoder_result_root
perl $transdecoder_root/TransDecoder.LongOrfs -t $canoncial_cDNA_path > $transdecoder_result_root/long_orf_record.log
perl $transdecoder_root/TransDecoder.Predict -t $canoncial_cDNA_path > $transdecoder_result_root/predict_record.log
