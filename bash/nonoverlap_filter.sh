#!/bin/bash
## function print usage
usage(){
 echo "Usage: Get nonoverlap region id"
 echo "  Arguments:"
 echo "    -i  <string>  Input path"
 echo "  Options:"
 echo "    -s  <bool>    Filter with same strand [default: false]"
 echo "    -e  <str>     Path to write overlapped id"
 echo "    -h  Print help message and exit"
 echo "Example: bash nonoverlap_filter.sh -i raw.bed -o nonoverlap_id.txt"
 echo ""
}

while getopts i:e:sh option
 do
  case "${option}"
  in
   i )input_path=$OPTARG;;
   e) error_id_path=$OPTARG;;
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

if [ ! "$use_strand" ]; then
    use_strand=false
fi

script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bed_file="${input_path%.*}"

if $use_strand ; then
    bash $script_root/sort_merge.sh -i $input_path -o ${bed_file}_sorted.bed -s
else    
    bash $script_root/sort_merge.sh -i $input_path -o ${bed_file}_sorted.bed
fi

awk -F '\t' -v OFS='\t' '{
    n = split($4, ids, ",")
    if(n==1)
    {
        print(ids[1])
    }
}' ${bed_file}_sorted.bed

if [ "$error_id_path" ]; then
    awk -F '\t' -v OFS='\t' '{
        n = split($4, ids, ",")
        if(n>1)
        {
            print(ids[1])
        }
    }' ${bed_file}_sorted.bed > $error_id_path
fi
rm ${bed_file}_sorted.bed
