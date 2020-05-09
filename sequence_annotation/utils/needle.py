import os

def get_value(lines,target):
    for line in lines:
        if line.startswith('# '+target):
            return float(line.split('(')[1].split('%')[0])
    raise

def get_identity(path):
    with open(path,'r') as fp:
        lines = fp.read().split('\n')
    return get_value(lines,'Identity')

def run_needle(seq1_path,seq2_path,output_path):
    COMMAND = 'needle -asequence {} -bsequence {} -auto -endweight -snucleotide1 -snucleotide2 --outfile {}'
    command = COMMAND.format(seq1_path,seq2_path,output_path)
    os.system(command)
    identity = get_identity(output_path)
    return identity

def main(predict_gff_path,answer_gff_path,region_table_path,saved_root,**kwargs):
    predict = read_gff(predict_path)
    answer = read_gff(answer_path)
    region_table = read_region_table(region_table_path)
    compare_and_save(predict,answer,region_table,saved_root,**kwargs)
