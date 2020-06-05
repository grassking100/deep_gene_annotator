import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_json
from sequence_annotation.preprocess.bed2gff import bed2gff
from sequence_annotation.preprocess.site_analysis import get_transcript_start_diff,get_transcript_end_diff,get_stats
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict

def coordinate_compare(answer,predict,round_value=None,**kwargs):
    start_diff = get_transcript_start_diff(answer, predict, **kwargs)
    end_diff = get_transcript_end_diff(answer, predict, **kwargs)
    site_diffs = {}
    site_diffs['TSS'] = get_stats(start_diff,round_value=round_value)
    site_diffs['cleavage_site'] = get_stats(end_diff,round_value=round_value)
    return site_diffs

def main(reference_bed_path,redefined_bed_path,output_path,id_table_path):
    
    id_table = get_id_convert_dict(id_table_path)
    reference = read_bed(reference_bed_path)
    redefined = read_bed(redefined_bed_path)
    reference = bed2gff(reference,id_table)
    redefined = bed2gff(redefined,id_table)
    
    result = {}
    result['abs_reference_as_ref'] = coordinate_compare(reference,redefined)
    result['abs_redefined_as_ref'] = coordinate_compare(reference,redefined,answer_as_ref=False)
    result['reference_as_ref'] = coordinate_compare(reference,redefined,absolute=False)
    result['redefined_as_ref'] = coordinate_compare(reference,redefined,absolute=False,answer_as_ref=False)
    write_json(result,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will compare tss and cs between two files")
    parser.add_argument("-r", "--reference_bed_path",help="Bed file to compared",required=True)
    parser.add_argument("-c", "--redefined_bed_path",help="Bed file to comparing",required=True)
    parser.add_argument("-o", "--output_path",help="Path of compared result",required=True)
    parser.add_argument("-t", "--id_table_path",required=True)
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)

    