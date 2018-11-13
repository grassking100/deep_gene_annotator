fasta_file=$1
fasta_file=${fasta_file%.*}
sed  '/>/ s/^>/>Chr/g' "${fasta_file}.fasta" > "${fasta_file}_rename.fasta"