import os
from .utils import write_json

def get_value(lines,target):
    for line in lines:
        if line.startswith('# '+target):
            return int(line.split('/')[0])
    raise

def get_identity(path):
    with open(path,'r') as fp:
        lines = fp.read().split('\n')
    return get_value(lines,'Identity')

def run_needle(seq1_path,seq2_path,output_path):
    COMMAND = 'needle -asequence {} -bsequence {} -auto -snucleotide1 -snucleotide2 --outfile {}'
    command = COMMAND.format(seq1_path,seq2_path,output_path)
    os.system(command)
    identity = get_identity(output_path)
    return identity

def get_matched_percentage(source_path,target_path,output_path):
    source = read_fasta(source_path)
    if len(source)!=1:
        raise
    source_len = len(list(source.values())[0])
    identity = run_needle(seq1_path,seq2_path,output_path)
    matched_percentage = identity/source_len
    return matched_percentage

def main(matched_root,output_root):
    matched_table_path = os.path.join(matched_root,'matched_table.tsv')
    matched_table = pd.read_csv(matched_table_path,sep='\t',index=None)
    matched_table = matched_table.to_dict('record')
    matched_percentage_result = {}
    for matched in matched_table:
        source_name = matched['source']
        target_name = matched['target']
        name = "{}_{}".format(source_name,target_name)
        source_path = os.path.join(matched_root,'source',"{}.fasta".format(source_name))
        target_path = os.path.join(matched_root,'target',"{}.fasta".format(target_name))
        output_path = os.path.join(output_root,'{}.needle'.format(name))
        matched_percentage = get_matched_percentage(source_path,target_path,output_path)
        matched_percentage_result[name] = matched_percentage
    write_json(matched_percentage_result,os.path.join(output_root,'matched_percentage.json'))
