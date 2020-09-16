import os,sys
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.file_process.utils import read_bed
from sequence_annotation.file_process.site_analysis import get_stats

def _coordinate_compare(compared,comparing):
    compared_group = compared.groupby('id')
    comparing_group = comparing.groupby('id')
    result = []
    for id_,compared_item in compared_group:
        if id_ not in comparing_group.groups.keys():
            result.append({'id':id_,'tss':float('nan'),'cleavage_site':float('nan')})
        else:
            comparing_item = comparing_group.get_group(id_)
            if len(compared_item)==1 and len(comparing_item)==1:
                strand = list(compared_item['strand'])[0]
                start_diff = list(comparing_item['start'])[0]-list(compared_item['start'])[0]
                end_diff = list(comparing_item['end'])[0]-list(compared_item['end'])[0]
                if strand == '+':
                    tss_diff = start_diff
                    cs_diff = end_diff
                else:
                    tss_diff = -end_diff
                    cs_diff = -start_diff
                result.append({'id':id_,'tss':tss_diff,'cleavage_site':cs_diff})
            else:
                raise Exception()
    return result
                
def coordinate_compare(reference,redefined):
    reference_as_ref = _coordinate_compare(reference,redefined)
    redefined_as_ref = _coordinate_compare(redefined,reference)
    result = {}
    for source,diff_data in zip(['reference','redefined'],[reference_as_ref,redefined_as_ref]):
        for type_ in ['tss','cleavage_site']:
            for to_abs in [False,True]:
                name = '{}_as_ref_{}'.format(source,type_)
                diffs = [item[type_] for item in diff_data]
                if to_abs:
                    diffs = np.abs(diffs)
                    name = 'abs_'+name
                result[name] = get_stats(diffs)
    return pd.DataFrame.from_dict(result)
    
def main(reference_bed_path,redefined_bed_path,output_path):
    reference = read_bed(reference_bed_path)
    redefined = read_bed(redefined_bed_path)
    coordinate_compare(reference,redefined).to_csv(output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will compare tss and cs between two files")
    parser.add_argument("-r", "--reference_bed_path",help="Bed file to compared",required=True)
    parser.add_argument("-c", "--redefined_bed_path",help="Bed file to comparing",required=True)
    parser.add_argument("-o", "--output_path",help="Path of compared result",required=True)
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
