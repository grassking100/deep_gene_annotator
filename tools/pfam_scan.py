import os
import sys
import pandas as pd
from argparse import ArgumentParser
from multiprocessing import Pool,cpu_count
sys.path.append(os.path.dirname(__file__) + "/..")
from sequence_annotation.utils.utils import read_fasta,create_folder,write_fasta,write_json

def pfam_scan(pfam_scan_path,fasta_path,db_root,output_path):
    command = "perl {} -fasta {} --dir {} -outfile {} -cpu 1"
    command = command.format(pfam_scan_path,fasta_path,db_root,output_path)
    print(command)
    os.system(command)
    try:
        df = pd.read_csv(output_path,comment='#',sep='\s+',header=None,engine='python')
        ids = set(df[0])
        return len(ids)
    except:
        return 0

def run_pfam_scan(pfam_scan_path,db_root,output_root,id_):
    fasta_path = os.path.join(output_root,"split","{}.fasta".format(id_))
    output_path = os.path.join(output_root,"result","{}.txt".format(id_))
    result = pfam_scan(pfam_scan_path,fasta_path,db_root,output_path)
    return id_,result

def translate(input_path,output_path):
    command='transeq -frame F {} {}'
    command = command.format(input_path,output_path)
    print(command)
    os.system(command)

def main(fasta_path,db_root,output_root,hmmer_src_root,pfam_scan_root):
    os.environ["PATH"] = hmmer_src_root+":"+os.environ["PATH"]
    os.environ["PERL5LIB"] = pfam_scan_root
    create_folder(output_root)
    create_folder(os.path.join(output_root,"split"))
    create_folder(os.path.join(output_root,"result"))
    pfam_scan_path=os.path.join(pfam_scan_root,'pfam_scan.pl')
    orfs = {}
    
    translated_path = os.path.join(output_root,"translated.fasta")
    translate(fasta_path,translated_path)
    translated = read_fasta(translated_path)
    for key,seq in translated.items():
        source ='_'.join(key.split('_')[:-1])
        if source not in orfs:
            orfs[source] = {}
        orfs[source][key]=seq

    kwargs = []
    for id_,fasta in orfs.items():
        path = os.path.join(output_root,"split","{}.fasta".format(id_))
        kwargs.append((pfam_scan_path,db_root,output_root,id_))
        write_fasta(fasta,path)

    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(run_pfam_scan, kwargs)

    result_df = pd.DataFrame.from_dict(results)
    result_df.columns=['id','count']
    result_df = result_df.sort_values('id')
    result_df.to_csv(os.path.join(output_root,'result.tsv'),sep='\t',index=None)
    origin_num = len(read_fasta(fasta_path))
    has_orf_num = len(orfs)
    matched_num = len(result_df[result_df['count']!=0])
    num = {'origin_num':origin_num,'has_orf_num':has_orf_num,'matched_num':matched_num}
    write_json(num,os.path.join(output_root,'count.json'))
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f","--fasta_path", required=True)
    parser.add_argument("-d","--db_root", required=True)
    parser.add_argument("-o","--output_root", required=True)
    parser.add_argument("--hmmer_src_root", required=True)
    parser.add_argument("--pfam_scan_root", required=True)
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
