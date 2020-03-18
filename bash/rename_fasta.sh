input_file=$1
input_file=${input_file%.*}
sed  '/>/ s/^>Chr/>/g' "${input_file}.fasta"
