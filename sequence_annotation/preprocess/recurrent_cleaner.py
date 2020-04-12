import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_bed, write_bed
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict
from sequence_annotation.preprocess.utils import classify_data_by_id

work_dir = "/".join(sys.argv[0].split('/')[:-1])
BASH_ROOT = "{}/../../bash".format(work_dir)
SAFE_FILTER_BASH_PATH = os.path.join(BASH_ROOT, 'safe_filter.sh')

if __name__ == "__main__":
    # Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-b", "--background_bed_path", required=True)
    parser.add_argument("-c", "--coordinate_consist_bed_path", required=True)
    parser.add_argument("-x", "--filtered_bed_path", required=True)
    parser.add_argument("-f", "--fai_path", required=True)
    parser.add_argument("-u", "--upstream_dist", type=int, required=True)
    parser.add_argument("-d", "--downstream_dist", type=int, required=True)
    parser.add_argument("-s", "--saved_root", required=True)
    parser.add_argument("-t", "--id_table_path")
    parser.add_argument("--use_strand", action='store_true')
    args = parser.parse_args()
    output_path = os.path.join(args.saved_root, "recurrent_cleaned.bed")
    id_path = 'region_upstream_{}_downstream_{}_safe_zone_id_{}.txt'
    if args.use_strand:
        safe_filter = "bash {} -i {} -x {} -f {} -u {} -d {} -o {} -s"
    else:
        safe_filter = "bash {} -i {} -x {} -f {} -u {} -d {} -o {}"
    if os.path.exists(output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        ###Read data###
        id_convert_dict = None
        if args.id_table_path is not None:
            id_convert_dict = get_id_convert_dict(args.id_table_path)
        background_bed = read_bed(args.background_bed_path)
        coordinate_consist_bed = read_bed(args.coordinate_consist_bed_path)
        filtered_bed = read_bed(args.filtered_bed_path)
        full_background = pd.concat(
            [background_bed, coordinate_consist_bed]).drop_duplicates()
        want_id = set(filtered_bed['id'])
        left_bed = full_background[~full_background['id'].isin(want_id)]
        all_bed = pd.concat([filtered_bed, left_bed]).reset_index(drop=True)
        want_bed_path = os.path.join(args.saved_root, 'want.bed')
        unwant_bed_path = os.path.join(args.saved_root, 'unwant.bed')
        saved_nums = []
        saved_num = -1
        index = 0
        ###Recurrent cleaning###
        while True:
            index += 1
            want_bed, unwant_bed = classify_data_by_id(
                all_bed, want_id, id_convert_dict)
            write_bed(want_bed, want_bed_path)
            write_bed(unwant_bed, unwant_bed_path)
            if len(want_bed) == 0:
                break
            id_path_ = os.path.join(args.saved_root,
                                    id_path.format(args.upstream_dist,
                                                   args.downstream_dist,
                                                   index))
            command = safe_filter.format(SAFE_FILTER_BASH_PATH, want_bed_path, unwant_bed_path,
                                         args.fai_path, args.upstream_dist,
                                         args.downstream_dist, id_path_)
            print("Execute command:" + command)
            os.system(command)
            want_id = [id_ for id_ in open(
                id_path_).read().split('\n') if id_ != '']
            num = len(want_id)
            print("Iter " + str(index) + " get number:" + str(num))
            saved_nums.append(num)
            if num == saved_num:
                break
            else:
                saved_num = num
                want_bed_path = os.path.join(
                    args.saved_root, 'want_iter_{}.bed'.format(index))
                unwant_bed_path = os.path.join(
                    args.saved_root, 'unwant_iter_{}.bed'.format(index))

        ###Write data###
        want_bed = filtered_bed[filtered_bed['id'].isin(want_id)]
        want_bed = want_bed[~ want_bed.duplicated()]
        write_bed(want_bed, output_path)
        with open(os.path.join(args.saved_root, 'recurrent_count.stats'), 'w') as fp:
            for index, num in enumerate(saved_nums):
                fp.write("Iterate " + str(index + 1) +
                         ":left " + str(num) + " transcript.\n")
