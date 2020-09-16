import os, sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder
from sequence_annotation.file_process.utils import read_bed, write_bed, BED_COLUMNS
from sequence_annotation.file_process.get_id_table import get_id_convert_dict


WORK_DIR = "/".join(sys.argv[0].split('/')[:-1])
BASH_ROOT = "{}/../../bash".format(WORK_DIR)
NONOVERLAP_BASH_PATH = os.path.join(BASH_ROOT, 'nonoverlap_filter.sh')

def simply_coord_with_gene_id(bed, id_convert=None):
    bed = bed[BED_COLUMNS[:6]]
    if id_convert is not None:
        gene_ids = [id_convert[id_] for id_ in bed['id']]
        bed = bed.assign(id=pd.Series(gene_ids).values)
    bed = bed.assign(score=pd.Series('.', index=bed.index))
    bed = bed.drop_duplicates()
    return bed


def nonoverlap_filter(bed,output_root,use_strand=False,id_convert_dict=None):
    create_folder(output_root)
    gene_bed_path = os.path.join(output_root, "gene_coord.bed")
    nonoverlap_id_path = os.path.join(output_root,"gene_coord_nonoverlap_id.txt")
    gene_bed = simply_coord_with_gene_id(bed,id_convert_dict)
    write_bed(gene_bed, gene_bed_path)
    if use_strand:
        command = "bash {} -i {} -s > {}"
    else:
        command = "bash {} -i {} > {}"
    command = command.format(NONOVERLAP_BASH_PATH, gene_bed_path,nonoverlap_id_path)
    print("Execute " + command)
    os.system(command)
    nonoverlap_id = list(pd.read_csv(nonoverlap_id_path,header=None)[0])#[id_ for id_ in open(nonoverlap_id_path).read().split('\n') if id_ != '']
    ###Write data###
    if id_convert_dict is not None:
        bed['gene_id'] = [id_convert_dict[id_] for id_ in bed['id']]
    else:
        bed['gene_id'] = bed['id']
    want_bed = bed[bed['gene_id'].isin(nonoverlap_id)]
    want_bed = want_bed[~want_bed.duplicated()]
    overlap_bed = bed[~bed['gene_id'].isin(nonoverlap_id)]
    overlap_bed = overlap_bed[~overlap_bed.duplicated()]
    return want_bed,overlap_bed


def main(input_bed_path,output_root,id_table_path=None,**kwargs):
    bed = read_bed(args.input_bed_path)
    id_convert_dict = None
    if id_table_path is not None:
        id_convert_dict = get_id_convert_dict(id_table_path)
    want_bed,overlap_bed = nonoverlap_filter(bed,output_root,id_convert_dict=id_convert_dict,**kwargs)
    output_path = os.path.join(output_root, "nonoverlap.bed")
    overlap_path = os.path.join(output_root, "overlap.bed")
    write_bed(want_bed, output_path)
    write_bed(overlap_bed, overlap_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_bed_path", required=True)
    parser.add_argument("-o", "--output_root", required=True)
    parser.add_argument("-t", "--id_table_path", required=False)
    parser.add_argument("--use_strand", action='store_true', required=False)
    args = parser.parse_args()
    main(**vars(args))
