import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed, write_bed
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict
from sequence_annotation.preprocess.utils import simply_coord_with_gene_id

work_dir = "/".join(sys.argv[0].split('/')[:-1])
BASH_ROOT = "{}/../../bash".format(work_dir)
NONOVERLAP_BASH_PATH = os.path.join(BASH_ROOT, 'nonoverlap_filter.sh')

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--coordinate_consist_bed_path", required=True)
    parser.add_argument("-s", "--saved_root", required=True)
    parser.add_argument("-t", "--id_table_path", required=False)
    parser.add_argument("--use_strand", action='store_true', required=False)
    args = parser.parse_args()
    output_path = os.path.join(args.saved_root, "nonoverlap.bed")
    overlap_path = os.path.join(args.saved_root, "overlap.bed")
    gene_bed_path = os.path.join(args.saved_root, "gene_coord.bed")
    nonoverlap_id_path = os.path.join(args.saved_root,
                                      "gene_coord_nonoverlap_id.txt")
    if os.path.exists(output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        ###Read data###
        id_convert_dict = None
        if args.id_table_path is not None:
            id_convert_dict = get_id_convert_dict(args.id_table_path)

        coordinate_consist_bed = read_bed(args.coordinate_consist_bed_path)
        gene_bed = simply_coord_with_gene_id(coordinate_consist_bed,
                                             id_convert_dict)
        write_bed(gene_bed, gene_bed_path)
        if args.use_strand:
            command = "bash {} -i {} -s > {}"
        else:
            command = "bash {} -i {} > {}"
        command = command.format(NONOVERLAP_BASH_PATH, gene_bed_path,
                                 nonoverlap_id_path)
        print("Execute " + command)
        os.system(command)
        nonoverlap_id = [
            id_ for id_ in open(nonoverlap_id_path).read().split('\n')
            if id_ != ''
        ]
        ###Write data###
        if id_convert_dict is not None:
            coordinate_consist_bed['gene_id'] = [
                id_convert_dict[id_] for id_ in coordinate_consist_bed['id']
            ]
        else:
            coordinate_consist_bed['gene_id'] = coordinate_consist_bed['id']
        want_bed = coordinate_consist_bed[
            coordinate_consist_bed['gene_id'].isin(nonoverlap_id)]
        want_bed = want_bed[~want_bed.duplicated()]
        write_bed(want_bed, output_path)

        overlap_bed = coordinate_consist_bed[
            ~coordinate_consist_bed['gene_id'].isin(nonoverlap_id)]
        overlap_bed = overlap_bed[~overlap_bed.duplicated()]
        write_bed(overlap_bed, overlap_path)
