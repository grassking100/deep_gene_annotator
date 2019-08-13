#!/bin/bash
## function print usage
usage(){
 echo "Usage: Sort and merged bed of column 4, 5, and 6"
 echo "  Arguments:"
 echo "    -i  <string>  Input path"
 echo "    -o  <string>  Output path"
 echo "  Options:"
 echo "    -s  <bool>    Merge with same strand [default: false]"
 echo "    -h  Print help message and exit"
 echo "Example: bash sort_merge.sh -i raw.bed -o result.bed"
 echo ""
}
while getopts i:o:sh option
 do
  case "${option}"
  in
   i )input_path=$OPTARG;;
   o )output_path=$OPTARG;;
   s )use_strand=true;;
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
if [ ! "$use_strand" ]; then
    use_strand=false
fi
num=$( cat $input_path | wc -l)
if (($num>0)) ;then
    if $use_strand ; then
        bedtools sort -i $input_path | bedtools merge -s -c 4,5,6 -o distinct,distinct,distinct \
        -delim ","  | awk -F'\t' -v OFS="\t"  '{print($1,$2,$3,$5,$6,$7)}' > $output_path
    else
        bedtools sort -i $input_path | bedtools merge -c 4,5,6 -o distinct,distinct,distinct -delim ","  > $output_path
    fi
else
    cp $input_path $output_path
fi

