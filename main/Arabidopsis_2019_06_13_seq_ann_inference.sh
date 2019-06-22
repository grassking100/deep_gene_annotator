script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python_script_path=$script_root/Arabidopsis_2019_06_13_seq_ann_inference.py
fasta_path='./io/Arabidopsis_thaliana/data/2019_06_13/fasta/selected_region.fasta'
ann_seqs_path='./io/Arabidopsis_thaliana/data/2019_06_13/result/selected_region.h5'
max_len=10000
test_chroms='5'
saved_root='./io/record/arabidopsis_2019_06_13'
mkdir $saved_root
mkdir $saved_root/seq_ann_inference
nohup python3 $python_script_path 0 $fasta_path $ann_seqs_path $max_len '1,2,3' '4' $test_chroms $saved_root/seq_ann_inference/fold_1 &

nohup python3 $python_script_path 1 $fasta_path $ann_seqs_path $max_len '2,3,4' '1' $test_chroms $saved_root/seq_ann_inference/fold_2 &

nohup python3 $python_script_path 2 $fasta_path $ann_seqs_path $max_len '3,4,1' '2' $test_chroms $saved_root/seq_ann_inference/fold_3 &

nohup python3 $python_script_path 3 $fasta_path $ann_seqs_path $max_len '4,1,2' '3' $test_chroms $saved_root/seq_ann_inference/fold_4 &