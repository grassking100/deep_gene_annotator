#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating annotation data"
 echo "  Arguments:"
 echo "    -u  <int>     Upstream distance"
 echo "    -w  <int>     Downstream distance"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -i  <string>  Path of bed"
 echo "    -o  <string>  Directory for output"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -t  <int>     Radius of Transcription start sites                [default: 100]"
 echo "    -d  <int>     Radius of Donor sites                              [default: 100]"
 echo "    -a  <int>     Radius of Accept sites                             [default: 100]"
 echo "    -c  <int>     Radius of Cleavage sites                           [default: 100]"
 echo "    -f  <bool>    Filter with ORF                                    [default: false]"
 echo "    -p  <string>  Gene and mRNA id conver path,it will be created if it doesn't be provided"
 echo "    -b  <string>  Path of background bed, it will be set to input path if it doesn't be provided"
 echo "    -h            Print help message and exit"
 echo "Example: bash process_data.sh -u 10000 -w 10000 -g /home/io/genome.fasta -i /home/io/example.bed -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:w:g:i:o:t:c:d:a:s:f:p:b:h option
 do
  case "${option}"
  in
   u )upstream_dist=$OPTARG;;
   w )downstream_dist=$OPTARG;;
   g )genome_path=$OPTARG;;
   i )bed_path=$OPTARG;;
   o )saved_root=$OPTARG;;
   t )tss_radius=$OPTARG;;
   c )cleavage_radius=$OPTARG;;
   d )donor_radius=$OPTARG;;
   a )accept_radius=$OPTARG;;
   s )source_name=$OPTARG;;
   f )filter_orf=$OPTARG;;
   p )id_convert_table=$OPTARG;;
   b )background_bed_path=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$upstream_dist" ]; then
    echo "Missing option -u"
    usage
    exit 1
fi
if [ ! "$downstream_dist" ]; then
    echo "Missing option -w"
    usage
    exit 1
fi
if [ ! "$genome_path" ]; then
    echo "Missing option -g"
    usage
    exit 1
fi
if [ ! "$bed_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi
if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi
if [ ! "$source_name" ]; then
    echo "Missing option -s"
    usage
    exit 1
fi

if [ ! "$tss_radius" ]; then
    tss_radius="100"
fi
if [ ! "$cleavage_radius" ]; then
    cleavage_radius="100"
fi
if [ ! "$donor_radius" ]; then
    donor_radius="100"
fi
if [ ! "$accept_radius" ]; then
    accept_radius="100"
fi
if [ ! "$filter_orf" ]; then
    filter_orf=false
fi
if [ ! "$background_bed_path" ]; then
    background_bed_path=$bed_path
fi
if [ ! "$create_id" ]; then
    create_id=false
fi

result_root=$saved_root/result
cleaning_root=$saved_root/cleaning
fasta_root=$saved_root/fasta
stats_root=$saved_root/stats
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
gene_info_root=$script_root/sequence_annotation/gene_info
genome_handler_root=$script_root/sequence_annotation/genome_handler

genome_fai=$genome_path.fai
echo "Start process_main.sh"
#Create folder
echo "Step 1: Create folder"
mkdir -p $saved_root
mkdir -p $result_root
mkdir -p $cleaning_root
mkdir -p $fasta_root
mkdir -p $stats_root

samtools faidx $genome_path
cp $bed_path $saved_root/data.bed


echo "Step 2: Remove overlapped data"
if [ ! "$id_convert_table" ]; then
    id_convert_table=$saved_root/id_table.tsv
    python3 $gene_info_root/create_id_table_by_coord.py -b $saved_root/data.bed -t $id_convert_table -p gene
    
fi

python3 $gene_info_root/nonoverlap_filter.py -c $saved_root/data.bed -i $id_convert_table -s $cleaning_root --use_strand

echo "Step 3: Remove overlap data with specific distance"
python3 $gene_info_root/recurrent_cleaner.py -r $background_bed_path -c $cleaning_root/nonoverlap.bed -f $genome_fai \
-u $upstream_dist -d $downstream_dist -s $cleaning_root -i $id_convert_table > $cleaning_root/recurrent.log


echo "Step 4: Create data around selected region"
cp $cleaning_root/recurrent_cleaned.bed $result_root/cleaned.bed

if  $filter_orf ; then
    #Create canonical data
    orf_clean_root=$saved_root/orf_clean
    mkdir -p $orf_clean_root
    cp $result_root/cleaned.bed $orf_clean_root/cleaned.bed
    echo -e 'id\tstart\tend' > $orf_clean_root/orf.tsv
    awk -F'\t' -v OFS="\t" '{print($4,$7+1,$8)}' $orf_clean_root/cleaned.bed >> $orf_clean_root/orf.tsv
    if [ ! -e "$orf_clean_root/alt_region.gff" ]; then
        echo "Canonical path decoding"
        python3 $gene_info_root/path_decode.py -i $orf_clean_root/cleaned.bed -o $orf_clean_root/alt_region.gff -t $id_convert_table
    fi
    #Get valid coding mRNA    
    python3 $gene_info_root/create_canonical_gff.py -i $orf_clean_root/alt_region.gff -r $orf_clean_root/orf.tsv \
    -t $id_convert_table -o $orf_clean_root/canonical.gff 2> $orf_clean_root/invalid_canonical.log
    
    python3 $gene_info_root/gff2bed.py -i $orf_clean_root/canonical.gff -o $orf_clean_root/canonical.bed
    
    python3 $gene_info_root/get_CDS_bed.py -i $orf_clean_root/canonical.bed -o $orf_clean_root/canonical_cds.bed 2> $orf_clean_root/no_CDS.log
    
    bedtools getfasta -name -split -s -fi $genome_path -bed $orf_clean_root/canonical_cds.bed -fo $orf_clean_root/canonical_cds.fasta
    python3 $gene_info_root/translate.py -i $orf_clean_root/canonical_cds.fasta \
    -t $gene_info_root/standard_codon.tsv -o $orf_clean_root/canonical_peptide.fasta --error_status_path $orf_clean_root/error_status.tsv \
    --valid_ids_path $orf_clean_root/valid_orf.id --valid_premature_stop_aa --valid_start_aa 'M' --valid_stop_aa '*'
    
    valid_num=$(wc -l < $orf_clean_root/valid_orf.id )
    cleaned_num=$(wc -l < $orf_clean_root/cleaned.bed )
    if (( $cleaned_num != $valid_num )); then
        python3 $gene_info_root/get_subbed.py -i $orf_clean_root/cleaned.bed -d $orf_clean_root/valid_orf.id \
        -t $id_convert_table -o $orf_clean_root/valid_orf.bed
        echo "Execute second recurrent cleaner"
        orf_recurrent=$orf_clean_root/orf_recurrent
        mkdir -p $orf_recurrent
        
        python3 $gene_info_root/recurrent_cleaner.py -r $background_bed_path -c $orf_clean_root/valid_orf.bed -f $genome_fai \
        -u $upstream_dist -d $downstream_dist -s $orf_recurrent -i $id_convert_table > $cleaning_root/recurrent.log
        cp $orf_recurrent/recurrent_cleaned.bed $result_root/cleaned.bed
        echo "End of second recurrent cleaner"

    fi
    
fi

if [ -e "$result_root/selected_region.fasta.fai" ]; then
    rm $result_root/selected_region.fasta.fai
fi
#Create region around mRNAs
bash $bash_root/get_region.sh -i $result_root/cleaned.bed -f $genome_fai -u $upstream_dist \
-d $downstream_dist -o $result_root/selected_region.bed -r

#get mRNAs in valid region
bash $bash_root/get_ids.sh -i $result_root/selected_region.bed > $result_root/valid_mRNA.id

python3 $gene_info_root/get_subbed.py -i $orf_clean_root/cleaned.bed -d $result_root/valid_mRNA.id -o $result_root/cleaned.bed

#Rename region
python3 $gene_info_root/rename_bed.py -i $result_root/selected_region.bed -p region \
-t $result_root/region_rename_table.tsv -o $result_root/selected_region.bed

#Get region with 5' --> 3' direction
bedtools getfasta -s -name -fi $genome_path -bed $result_root/selected_region.bed -fo $result_root/selected_region.fasta
samtools faidx $result_root/selected_region.fasta

#Get mRNAs with 5' --> 3' direction in region
python3 $gene_info_root/redefine_coordinate.py -i $result_root/cleaned.bed -t $result_root/region_rename_table.tsv \
-r $result_root/selected_region.bed -o $result_root/cleaned.bed

if [ ! -e "$result_root/alt_region.gff" ]; then
    echo "Canonical path decoding"
    python3 $gene_info_root/path_decode.py -i $result_root/cleaned.bed -o $result_root/alt_region.gff -t $id_convert_table
fi

#Get ORF region of mRNAs
echo -e 'id\tstart\tend' > $result_root/orf.tsv
awk -F'\t' -v OFS="\t"  '{print($4,$7+1,$8)}' $result_root/cleaned.bed >> $result_root/orf.tsv

#Create canonical gff
python3 $gene_info_root/create_canonical_gff.py -i $result_root/alt_region.gff -r $result_root/orf.tsv \
-t $id_convert_table -o $result_root/canonical.gff 2> $result_root/invalid_canonical.log
if  $filter_orf ; then
    if [ ! -e "$result_root/canonical.h5" ]; then
        python3 $genome_handler_root/alt_anns_creator.py -i $result_root/canonical.gff \
        -f $result_root/selected_region.fasta.fai -o $result_root/canonical.h5 -s source_name
    fi
fi
if [ ! -e "$result_root/alt_region.h5" ]; then
    python3 $genome_handler_root/alt_anns_creator.py -i $result_root/alt_region.gff \
    -f $result_root/selected_region.fasta.fai -o $result_root/alt_region.h5 -s source_name
fi

echo "Step 5: Get fasta of region around TSSs, CAs, donor sites, accept sites and get fasta of peptide and cDNA"
echo "       , and write statistic data"
python3 $gene_info_root/gff2bed.py -i $result_root/canonical.gff -o $result_root/canonical.bed

bash $bash_root/bed_analysis.sh -i $result_root/canonical.bed -g $result_root/selected_region.fasta -o $fasta_root -s $stats_root

awk -F'\t' -v OFS="\t"  '{if($3=="exon"){print($5-$4+1)}}' $result_root/canonical.gff > $stats_root/exon.length
awk -F'\t' -v OFS="\t"  '{if($3=="intron"){print($5-$4+1)}}' $result_root/canonical.gff > $stats_root/intron.length
awk -F'\t' -v OFS="\t"  '{if($3=="UTR"){print($5-$4+1)}}' $result_root/canonical.gff > $stats_root/utr.length
awk -F'\t' -v OFS="\t"  '{if($3=="CDS"){print($5-$4+1)}}' $result_root/canonical.gff > $stats_root/cds.length

num_input_mRNAs=$(wc -l < $bed_path )
num_background_mRNAs=$(wc -l < $background_bed_path )
num_nonoverlap=$(wc -l < $cleaning_root/nonoverlap.bed )
num_recurrent=$(wc -l < $cleaning_root/recurrent_cleaned.bed )
num_cleaned=$(wc -l < $result_root/cleaned.bed )
num_region=$(wc -l < $result_root/selected_region.bed )

echo "The number of background mRNAs: $num_background_mRNAs" > $stats_root/count.stats
echo "The number of selected to input mRNAs: $num_input_mRNAs" >> $stats_root/count.stats
echo "The number of mRNAs which are not overlap with each other: $num_nonoverlap" >> $stats_root/count.stats
echo "The number of recurrent-cleaned mRNAs: $num_recurrent" >> $stats_root/count.stats


if  $filter_orf ; then
    canonical_num=$(wc -l < $orf_clean_root/valid_orf.id )
    no_CDS_num=$(($(wc -l < $orf_clean_root/no_CDS.log )/2))
    invalid_canonical_num=$(($(wc -l < $orf_clean_root/invalid_canonical.log )/2))
    error_status_num=$(($(wc -l < $orf_clean_root/error_status.tsv) -1))
    invalid_canonical_num=$(( $no_CDS_num + $invalid_canonical_num + $error_status_num))
    valid_orf_left=$(wc -l < $orf_clean_root/valid_orf.bed )
    orf_recurrent_left=$(wc -l < $orf_recurrent/recurrent_cleaned.bed )
    echo "The number of valid canonical gene: $canonical_num" >> $stats_root/count.stats
    echo "The number of invalid canonical gene: $invalid_canonical_num" >> $stats_root/count.stats
    echo "The number of valid ORF and canonical mRNA: $valid_orf_left" >> $stats_root/count.stats
    echo "The number of valid ORF and recurrent cleaned canonical mRNA: $orf_recurrent_left" >> $stats_root/count.stats
fi

echo "The number of cleaned mRNAs in valid region: $num_cleaned" >> $stats_root/count.stats
echo "The number of regions: $num_region" >> $stats_root/count.stats

echo "End process_main.sh"
exit 0
