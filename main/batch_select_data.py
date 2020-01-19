import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from sequence_annotation.utils.utils import create_folder,write_json,read_json
from main.select_data import main as select_data_main

def _get_name(path):
    return path.split('/')[-1].split('.')[0]
    
def main(saved_root,usage_table_path=None,stats_path=None,max_len=None,**kwargs):
    usage_table = pd.read_csv(usage_table_path)
    
    usage_table = usage_table.to_dict('record')

    if stats_path is not None:
        max_len = round(read_json(stats_path)['MAD derived threshold'])
        print(max_len)

    new_table=[]
    for item in usage_table:
        paths = {}
        for key,path in item.items():
            file_name = '{}.h5'.format(_get_name(path))
            saved_path = os.path.join(saved_root,file_name)
            paths[key] = saved_path
            select_data_main(id_path=path,saved_path=saved_path,max_len=max_len,**kwargs)
        new_table.append(paths)

    new_table = pd.DataFrame.from_dict(new_table)
    path = os.path.join(saved_root,'data_usage_path.csv')
    new_table.to_csv(path,index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-a","--ann_seqs_path",help="Path of AnnSeqContainer",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-u","--usage_table_path",help="Usage table in csv format",required=True)
    parser.add_argument("--stats_path",help="The path of statistic result in json format,"+\
                        "its 'MAD derived threshold' would be set to max len")
    parser.add_argument("--max_len",type=int,default=None,help="Sequences' max length," +\
                        " if it is set by stats_path then it will be ignored")
    parser.add_argument("--min_len",type=int,default=0,help="Sequences' min length")
    parser.add_argument("--ratio",type=float,default=1,help="Ratio of number to be chosen to train" +\
                        " and validate, start chosen by increasing order)")
    parser.add_argument("--select_each_type",action='store_true')

    args = parser.parse_args()
    setting = vars(args)
    
    create_folder(args.saved_root)
    path = os.path.join(args.saved_root,"batch_select_data_config.json")
    write_json(setting,path)
    
    main(**setting)
