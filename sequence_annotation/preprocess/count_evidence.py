import sys
import os
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_gff,write_json

def _get_ids(gff):
    ids = list(gff['chr']+"_"+gff['start'].astype(str)+"_"+gff['strand'])
    return ids

def _get_unique_id_len(data):
    return len(set(_get_ids(data)))

def main(coordinate_redefined_root,output_path):
    external_five_UTR_tss = read_gff(os.path.join(coordinate_redefined_root,'external_five_UTR_tss.gff3'))
    external_three_UTR_cs = read_gff(os.path.join(coordinate_redefined_root,'external_three_UTR_cleavage_site.gff3'))

    safe_tss = read_gff(os.path.join(coordinate_redefined_root,'safe_tss.gff3'))
    safe_cs = read_gff(os.path.join(coordinate_redefined_root,'safe_cs.gff3'))

    external_five_UTR_tss_num = _get_unique_id_len(external_five_UTR_tss)
    external_three_UTR_cs_num = _get_unique_id_len(external_three_UTR_cs)
    safe_tss_num = _get_unique_id_len(safe_tss)
    safe_cs_num = _get_unique_id_len(safe_cs)
    stats = {
        'External 5\' UTR TSS number':external_five_UTR_tss_num,
        'Safe TSS number':safe_tss_num,
        'External 3\' UTR CS number':external_three_UTR_cs_num,
        'Safe CS number':safe_cs_num,
    }
    write_json(stats,output_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--coordinate_redefined_root", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()

    kwargs = vars(args)
    main(**kwargs)
