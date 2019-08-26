#!/bin/bash
## function print usage
usage(){
 echo "Usage: Print ids of BED file"
 echo "  Arguments:"
 echo "    -i  <string>  BED path"
 echo "  Options:"
 echo "    -h  Print help message and exit"
 echo "Example: bash get_ids.sh -i want.bed"
 echo ""
}
while getopts i:h option
 do
  case "${option}"
  in
   i )input_path=$OPTARG;;
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

awk -F '\t' -v OFS='\t' '{
    n = split($4, ids, ",")
    for(i=1;i<=n;i++)
    {
        print(ids[i])
    }
}' $input_path
