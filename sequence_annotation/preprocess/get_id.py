import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,read_gff,get_gff_with_attribute
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict

def main(input_path,output_path,id_table_path=None):
    file_format = input_path.split('.')[-1]
    if file_format in ['bed','bed12']:
        bed = read_bed(input_path)
        ids = set(bed['id'])
    elif file_format in ['gff','gff3']:
        gff = get_gff_with_attribute(read_gff(input_path))
        ids = set(gff['id'])
    else:
        raise Exception("Unknown file format {}".format(file_format))
    
    if id_table_path is not None:
        returned_ids = []
        id_convert_dict = get_id_convert_dict(id_table_path)
        for id_ in ids:
            returned_ids.append(id_convert_dict[id_])
    else:
        returned_ids = ids
    
    with open(output_path,"w") as fp:
        for id_ in returned_ids:
            fp.write("{}\n".format(id_))

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will output GFF id or BED id")
    parser.add_argument("-i", "--input_path",help="Input file",required=True)
    parser.add_argument("-o", "--output_path",help="Output id file",required=True)
    parser.add_argument("-t", "--id_table_path",help="Table about id conversion, "
                        "if it provided then the id would be converted by table")
    args = parser.parse_args()
    kwargs = vars(args)
    
    main(**kwargs)
