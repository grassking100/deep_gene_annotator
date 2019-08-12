#!/bin/bash
## function print usage
usage(){
 echo "Usage: Get upstream and donwstream region"
 echo "  Arguments:"
 echo "    -i  <string>  Input path"
 echo "    -o  <string>  Output path"
 echo "    -f  <string>  fai file path"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "  Options:"
 echo "    -r  <bool>    Use restrict mode to check length is perfect match or not [default: false]"
 echo "    -h  Print help message and exit"
 echo "Example: bash get_region.sh -i raw.bed -o result.bed -u 100 -d 100 -f genome.fai"
 echo ""
}
while getopts i:o:u:d:f:r:h option
 do
  case "${option}"
  in
   i )input_path=$OPTARG;;
   o )output_path=$OPTARG;;
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   f )fai_path=$OPTARG;;
   r )restrict_mode=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$input_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi
if [ ! "$output_path" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi
if [ ! "$upstream_dist" ]; then
    echo "Missing option -u"
    usage
    exit 1
fi
if [ ! "$downstream_dist" ]; then
    echo "Missing option -d"
    usage
    exit 1
fi
if [ ! "$fai_path" ]; then
    echo "Missing option -f"
    usage
    exit 1
fi

if [ ! "$restrict_mode" ]; then
    restrict_mode=false
fi

script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bedtools slop -s -i $input_path -g $fai_path -l $upstream_dist -r $downstream_dist > _extened.bed


if $restrict_mode ; then
    awk -F'\t' -v OFS="\t"  ' { print($4,$1,$2,$3,$5,$6)}' _extened.bed >  _extened.tsv
    awk -F'\t' -v OFS="\t"  ' { print($4,$1,$2,$3,$5,$6,$7,$8,$9,$10,$11,$12)}' $input_path >  _input.tsv
    join -j 1 -t $'\t' <(sort _extened.tsv) <(sort _input.tsv) | sort -n > _merged.tsv

    awk -F'\t' -v a="$upstream_dist" -v b="$downstream_dist" -v OFS="\t" '{   
                                 lhs_length=$4-$3
                                 rhs_length=$9-$8 + a + b
                                 if(lhs_length==rhs_length)
                                 {
                                     print($2,$3,$4,$1,$5,$6)
                                 }
      
                             }'  _merged.tsv > _consist.bed
    rm _extened.tsv
    rm _input.tsv
    rm _merged.tsv

else
    echo _extened.bed > _consist.bed
fi

bash $script_root/sort_merge.sh -i _consist.bed -o $output_path

rm _extened.bed
rm _consist.bed
