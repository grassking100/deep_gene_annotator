import os,sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_bed,write_bed
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict

def gene_score_filter(bed,threshold,mode,id_table_path):
    id_convert = get_id_convert_dict(id_table_path)
    bed['parent'] = [id_convert[id_] for id_ in bed['id']]
    bed_group = bed.groupby('parent')
    groups = []
    for gene_id,group in bed_group:
        if (group['score']!='.').all():
            if mode == 'bigger_or_equal':
                all_pass = (group['score'].astype(float)>=threshold).all()
            elif mode == 'smaller_or_equal':
                all_pass = (group['score'].astype(float)<=threshold).all()
            else:
                raise Exception()
            if all_pass:
                groups.append(group)
    bed = pd.concat(groups)
    return bed

def transcript_score_filter(bed,threshold,mode):
    has_score_index = bed['score']!='.'
    if mode == 'bigger_or_equal':
        pass_index = (bed.loc[has_score_index,'score'].astype(float)>=threshold).index
    elif mode == 'smaller_or_equal':
        pass_index = (bed.loc[has_score_index,'score'].astype(float)<=threshold).index
    else:
        raise Exception()
    bed = bed.loc[pass_index]
    return bed

def main(input_bed_path,threshold,mode,output_bed_path,remove_gene=False,id_table_path=None):
    bed = read_bed(input_bed_path)
    if remove_gene:
        if id_table_path is None:
            raise Exception("The id_table_path must be provided if the remove_gene is true')")
        bed = gene_score_filter(bed,threshold,mode,id_table_path)
    else:
        bed = transcript_score_filter(bed,threshold,mode)
    write_bed(bed,output_bed_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="Remove gene or transcript doesn't pass threshold filter")
    parser.add_argument("-i", "--input_bed_path",required=True)
    parser.add_argument("-o", "--output_bed_path",required=True)
    parser.add_argument("--threshold",type=float,required=True)
    parser.add_argument("--mode",choices=['bigger_or_equal', 'smaller_or_equal'],required=True)
    parser.add_argument("--remove_gene",action='store_true',help="Remove gene if at least one of its transcript doesn't pass threshold filter,"\
                       "otherwiese, only remove the transcript which doesn't pass threshold filter")
    parser.add_argument("-t","--id_table_path",type=str,help='The id_table_path must be provided if the remove_gene is true')
    args = parser.parse_args()
    main(**vars(args))