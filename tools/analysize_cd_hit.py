import sys,os
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/..")
from sequence_annotation.utils.utils import read_fasta, write_json,create_folder

def main(cdhit_result_path,output_root):
    create_folder(output_root)
    cluster_path = "{}.clstr".format(cdhit_result_path)
    with open(cluster_path,'r') as fp:
        data = fp.read()
    
    cluster = {}
    parent_ids = {}
    matched_ids = set()
    for line in data.split('\n'):
        if len(line)>0:
            if line.startswith('>'):
                cluster_id = line[1:]
                cluster[cluster_id] = -1
            else:
                cluster[cluster_id] += 1
                id_ = (line.split(',')[1].split('...')[0][2:])
                if cluster[cluster_id] == 0:
                    if id_ in parent_ids:
                        raise
                    parent_ids[cluster_id] = id_
                elif cluster[cluster_id] > 0:
                    matched_ids.add(id_)

    cluster = pd.DataFrame.from_dict([cluster],orient='columns').T
    cluster_count = cluster[0].value_counts()
    unmatched_ids = set(read_fasta(cdhit_result_path).keys())
    all_ids = unmatched_ids | matched_ids
    count = {
        'group':len(parent_ids),
        'hitted group':len(cluster[cluster[0]>0]),
        'unhitted group':len(cluster[cluster[0]==0]),
        'hit ratio':len(cluster[cluster[0]>0])/len(cluster),
        'matched ratio':len(matched_ids)/len(all_ids),
        'matched item':len(matched_ids),
        'unmatched item':len(unmatched_ids),
        'item':len(all_ids)
    }
    
    cluster_count.to_csv(os.path.join(output_root,'hit_number.csv'),index_label='hit_number',header=['count'])
    with open(os.path.join(output_root,'ancestor_ids.txt'),'w') as fp:
        for id_ in parent_ids.values():
            fp.write(id_+"\n")
    with open(os.path.join(output_root,'matched_ids.txt'),'w') as fp:
        for id_ in matched_ids:
            fp.write(id_+"\n")
    with open(os.path.join(output_root,'unmatched_ids.txt'),'w') as fp:
        for id_ in unmatched_ids:
            fp.write(id_+"\n")
    write_json(count,os.path.join(output_root,'count.json'))

    
if __name__ == '__main__':
    parser = ArgumentParser(description="Analysize cdhit result")
    parser.add_argument("-i","--cdhit_result_path",required=True,
                        help="Root of cdhit result")
    parser.add_argument("-o","--output_root",required=True)
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
