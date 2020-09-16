import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import read_bed, write_bed
from sequence_annotation.file_process.get_id_table import get_id_convert_dict

def safe_filter(want_path,unwant_path,fai_path,
                upstream_dist,downstream_dist,
                output_root,use_strand=False):
    expand_path=os.path.join(output_root,"region_up_{}_down_{}.bed".format(upstream_dist,downstream_dist))
    safe_path=os.path.join(output_root,"region_up_{}_down_{}_safe_zone.bed".format(upstream_dist,downstream_dist))
    #echo Expand around $want_path
    command = "bedtools slop -s -i {} -g {} -l {} -r {} > {}".format(want_path,fai_path,upstream_dist,downstream_dist,expand_path)
    os.system(command)
    #Output safe zone
    if use_strand:
        command = "bedtools intersect -s -a {} -b {} -wa -v > {}".format(expand_path,unwant_path,safe_path)
    else:
        command = "bedtools intersect -a {} -b {} -wa -v > {}".format(expand_path,unwant_path,safe_path)
    os.system(command)
    ids = list(read_bed(safe_path)['id'])
    os.system("rm {}".format(expand_path))
    os.system("rm {}".format(safe_path))
    #echo Get id
    return ids

def classify_data_by_id(bed, selected_ids, id_convert=None):
    all_ids = list(set(bed['id']))
    selected_ids = set(selected_ids)
    if id_convert is not None:
        selected_gene_ids = set(id_convert[id_] for id_ in selected_ids)
        id_info = pd.DataFrame(all_ids)
        id_info.columns = ['ref_id']
        gene_id = []
        for id_ in id_info['ref_id']:
            if id_ in id_convert.keys():
                gene_id.append(id_convert[id_])
            else:
                gene_id.append(id_)
        id_info = id_info.assign(gene_id=pd.Series(gene_id).values)
        match_status = id_info['gene_id'].isin(selected_gene_ids)
        want_status = id_info['ref_id'].isin(selected_ids)
        id_info['status'] = 'unwant'
        id_info.loc[match_status & want_status, 'status'] = 'want'
        id_info.loc[match_status & ~want_status, 'status'] = 'discard'
        want_id = id_info[id_info['status'] == 'want']['ref_id']
        unwant_id = id_info[id_info['status'] == 'unwant']['ref_id']
    else:
        id_info = pd.DataFrame(all_ids, columns=['id'])
        want_status = id_info['id'].isin(selected_ids)
        want_id = id_info[want_status]['id']
        unwant_id = id_info[~want_status]['id']
    want_bed = bed[bed['id'].isin(want_id)].drop_duplicates()
    unwant_bed = bed[bed['id'].isin(unwant_id)].drop_duplicates()
    return want_bed, unwant_bed

def recurrent_cleaner(input_bed,background_bed,id_convert_dict,genome_path,
                      upstream_dist,downstream_dist,output_root,use_strand=False):
    background_bed = background_bed.drop_duplicates()
    output_path = os.path.join(output_root, "recurrent_cleaned.bed")
    want_bed_path = os.path.join(output_root, 'want.bed')
    unwant_bed_path = os.path.join(output_root, 'unwant.bed')
    want_id = set(input_bed['id'])
    #Get transcript which its transcript id is not in filtered bed
    left_bed = background_bed[~background_bed['id'].isin(want_id)]
    all_bed = pd.concat([input_bed, left_bed]).reset_index(drop=True)
    saved_nums = []
    saved_num = -1
    index = 0
    ###Recurrent cleaning###
    while True:
        index += 1
        #Get classified transcripts by its gene id
        want_bed, unwant_bed = classify_data_by_id(all_bed, want_id,id_convert_dict)
        write_bed(want_bed, want_bed_path)
        write_bed(unwant_bed, unwant_bed_path)
        if len(want_bed) == 0:
            break
        want_id = safe_filter(want_bed_path,unwant_bed_path,genome_path,
                              upstream_dist,downstream_dist,output_root,use_strand=use_strand)
        num = len(want_id)
        print("Iter " + str(index) + " get number:" + str(num))
        saved_nums.append(num)
        if num == saved_num:
            break
        else:
            saved_num = num
            want_bed_path = os.path.join(output_root, 'want_iter_{}.bed'.format(index))
            unwant_bed_path = os.path.join(output_root, 'unwant_iter_{}.bed'.format(index))

    ###Write data###
    want_bed = input_bed[input_bed['id'].isin(want_id)]
    want_bed = want_bed[~ want_bed.duplicated()]
    write_bed(want_bed, output_path)
    with open(os.path.join(output_root, 'recurrent_count.stats'), 'w') as fp:
        for index, num in enumerate(saved_nums):
            fp.write("Iterate {}: left {} transcript.\n".format(index + 1,num))
    return want_bed

def main(input_bed_path,background_bed_path,id_table_path,**kwargs):
    input_bed = read_bed(input_bed_path)
    background_bed = read_bed(background_bed_path)
    id_convert_dict = get_id_convert_dict(id_table_path)
    recurrent_cleaner(input_bed,background_bed,id_convert_dict,**kwargs)
    
if __name__ == "__main__":
    # Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_bed_path", required=True)
    parser.add_argument("-b", "--background_bed_path", required=True)
    parser.add_argument("-f", "--genome_path", required=True)
    parser.add_argument("-u", "--upstream_dist", type=int, required=True)
    parser.add_argument("-d", "--downstream_dist", type=int, required=True)
    parser.add_argument("-o", "--output_root", required=True)
    parser.add_argument("-t", "--id_table_path", required=True)
    parser.add_argument("--use_strand", action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
