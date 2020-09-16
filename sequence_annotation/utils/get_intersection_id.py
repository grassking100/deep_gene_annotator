import os,sys
import venn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")

def intersection(ids_list,id_names,output_path=None,venn_path=None):
    if len(ids_list) != len(id_names):
        raise Exception("The numbers of data and names is not the same")
    data = {}
    intersection_ids = None
    sort_indice = np.argsort(id_names)
    id_names = [id_names[index] for index in sort_indice]
    ids_list = [ids_list[index] for index in sort_indice]
    for name,ids in zip(id_names,ids_list):
        ids = set(ids)
        data[name] = ids
        if intersection_ids is None:
            intersection_ids = ids
        else:
            intersection_ids = intersection_ids.intersection(ids)
            
    if output_path is not None:
        with open(output_path,'w') as fp:
            for id_ in  intersection_ids:
                fp.write("{}\n".format(id_))
        
    if venn_path is not None:
        venn.venn(data,fontsize=8,legend_loc="upper right")
        plt.savefig(venn_path)
        
    return intersection_ids
    
def main(paths,names,**kwargs):
    paths = paths.split(',')
    names = names.split(',')
    ids_list = []
    for path in paths:
        ids = set(pd.read_csv(path,header=None)[0])
        ids_list.append(ids)
    intersection(ids_list,names,**kwargs)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will output intersection ids")
    parser.add_argument("-i", "--input_paths",help="Input files",required=True)
    parser.add_argument("-n", "--id_names",help="The names of each data",required=True)
    parser.add_argument("-o", "--output_path",help="Output id file")
    parser.add_argument("-v", "--venn_path",help="The path of venn diagram")
    args = parser.parse_args()
    kwargs = vars(args)
    
    main(**kwargs)
