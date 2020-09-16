import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder


def main(trained_root, output_root, save_command_table_path,
         fasta_path, fasta_double_strand_path,
         revised_root,region_table_path):
    
    MAIN_PATH = os.path.join(os.path.dirname(__file__),'model_predict.py')
        
    COMMAND = "python3 " + MAIN_PATH + " -d {} -f {} -e {} -o {} -r {} -t {}"
    paths = [MAIN_PATH,fasta_path,fasta_double_strand_path,region_table_path]

    for path in paths:
        if not os.path.exists(path):
            raise Exception("{} is not exists".format(path))

    create_folder(output_root)
    
    with open(save_command_table_path, "w") as fp:
        fp.write("command\n")
        for name in sorted(os.listdir(trained_root)):
            
            path = os.path.join(trained_root,name)
            if name.startswith('.') or not os.path.isdir(path):
                continue

            result_root = os.path.join(output_root, name)    
            command = COMMAND.format(path, fasta_path, fasta_double_strand_path, 
                                     result_root,revised_root, region_table_path)
            command += "\n"
            fp.write(command)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--trained_root", required=True)
    parser.add_argument("-o", "--output_root", required=True)
    parser.add_argument("-c", "--save_command_table_path", required=True)
    parser.add_argument("-f","--fasta_path",help="Path of single-strand fasta",required=True)
    parser.add_argument("-e","--fasta_double_strand_path",help="Path of double-strand fasta",required=True)
    parser.add_argument("-r","--revised_root",help="The root of revised result",required=True)
    parser.add_argument("-t","--region_table_path",help="Path of region table",required=True)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
