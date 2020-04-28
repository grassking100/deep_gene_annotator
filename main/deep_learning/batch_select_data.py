import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/..")
from main.deep_learning.select_data import main as select_data_main
from sequence_annotation.utils.utils import create_folder, write_json

def _get_name(path, with_postfix=False):
    rel_path = path.split('/')[-1]
    if with_postfix:
        return rel_path
    else:
        return rel_path.split('.')[0]


def main(saved_root, usage_table_path=None,
         max_len=None, max_key=None, **kwargs):
    usage_table_root = '/'.join(usage_table_path.split('/')[:-1])
    usage_table = pd.read_csv(usage_table_path)
    usage_table = usage_table.to_dict('record')

    new_table = []
    for item in usage_table:
        paths = {}
        for key, path in item.items():
            path = os.path.join(
                usage_table_root, _get_name(
                    path, with_postfix=True))
            saved_rel_path = '{}.h5'.format(_get_name(path))
            saved_path = os.path.join(saved_root, saved_rel_path)
            paths[key] = saved_rel_path
            select_data_main(
                id_path=path,
                saved_path=saved_path,
                max_len=max_len,
                **kwargs)
        new_table.append(paths)

    new_table = pd.DataFrame.from_dict(new_table)
    path = os.path.join(saved_root, 'data_usage_path.csv')
    new_table.to_csv(path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--fasta_path", required=True,
                        help="Path of fasta")
    parser.add_argument("-a", "--ann_seqs_path", required=True,
                        help="Path of AnnSeqContainer")
    parser.add_argument("-s", "--saved_root", required=True,
                        help="Root to save file")
    parser.add_argument("-u", "--usage_table_path", required=True,
                        help="Usage table in csv format")
    parser.add_argument("--max_len", type=int, default=None, help="Sequences' max length")
    parser.add_argument("--min_len", type=int, default=0,
                        help="Sequences' min length")
    parser.add_argument("--ratio", type=float, default=1, help="Ratio of number to be chosen"
                        "to train and validate, start chosen by increasing order)")
    parser.add_argument("--select_each_type", action='store_true')
    parser.add_argument("--max_key")

    args = parser.parse_args()
    setting = vars(args)

    create_folder(args.saved_root)
    path = os.path.join(args.saved_root, "batch_select_data_config.json")
    write_json(setting, path)

    main(**setting)
