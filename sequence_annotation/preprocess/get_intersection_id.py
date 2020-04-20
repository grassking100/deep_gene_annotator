import os,sys
import pandas as pd
import venn
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_bed,read_gff,get_gff_with_attribute
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict

def main(input_paths,id_names,output_path=None,venn_path=None):
    input_paths = input_paths.split(',')
    id_names = id_names.split(',')

    if len(input_paths) != len(id_names):
        raise Exception("The numbers of paths and names is not the same")
    intersection_ids = None
    data = {}
    for name,path in zip(id_names,input_paths):
        try:
            ids = set(pd.read_csv(path,header=None)[0])
        except:
            raise Exception("Something wrong happens in {}".format(path))
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
