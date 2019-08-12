import os, sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.fasta  import read_fasta, write_fasta

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description='Translate seqeunce from first frame')
    parser.add_argument("-i","--input_path", help="cDNA Fasta path",required=True)
    parser.add_argument("-t","--translation_table_path", help="Translation table path",required=True)
    parser.add_argument("-o","--output_path", help="Output peptide path",required=True)
    parser.add_argument("--valid_ids_path", help="Path to write valid sequences' ids",required=True)
    parser.add_argument("--error_status_path", help="Path to write wrong sequences' ids and statuses",required=True)
    parser.add_argument("--not_show_stop_signal",help="Not show last stop signal",
                        required=False, action='store_true')
    parser.add_argument("--use_three_letters",help="Use three letters abbreviation instead of one",
                        required=False, action='store_true')
    parser.add_argument("--valid_premature_stop_aa",required=False, action='store_true')
    parser.add_argument("--valid_start_aa",required=False)
    parser.add_argument("--valid_stop_aa",required=False)
    args = parser.parse_args()
    df = pd.read_csv(args.translation_table_path,sep='\t',header=None,comment='#')
    if args.use_three_letters:
        trans_table = dict(zip(df[0],df[2]))
    else:    
        trans_table = dict(zip(df[0],df[1]))
    cDNAs = read_fasta(args.input_path)
    peptides = {}
    invalid_status = {}
    error_ids = []
    start_aa = args.valid_start_aa or 'M'
    stop_aa = args.valid_stop_aa or '*'
    
    for id_,cDNA in cDNAs.items():
        invalid_status[id_] = []
        valid = True
        length = len(cDNA)
        peptide = []
        if length%3 == 0:
            for start_index in range(0,length,3):
                codon = ''.join(cDNA[start_index:start_index+3]).upper()
                try:
                    peptide.append(trans_table[codon])
                except:
                    raise Exception("{} in {} is not in translation table".format(codon,id_))
            if args.valid_start_aa is not None and start_aa != peptide[0]:
                valid = False
                invalid_status[id_].append('wrong start codon')
            if args.valid_start_aa is not None and stop_aa != peptide[-1]:
                valid = False
                invalid_status[id_].append('wrong stop codon')
            if args.valid_premature_stop_aa and stop_aa in peptide[:-1]:
                valid = False
                invalid_status[id_].append('premature stop codon')
        else:
            valid = False
            invalid_status[id_].append("length is not mulitple of three")
        if valid:
            if args.not_show_stop_signal:
                peptide = ''.join(peptide[:-1])
                
            else:
                peptide = ''.join(peptide)
            peptides[id_] = peptide
        else:
            error_ids.append(id_)

    write_fasta(args.output_path,peptides)
    with open(args.error_status_path,"w") as fp:
        fp.write("id\terror status\n")
        for id_ in error_ids:
            status = ', '.join(invalid_status[id_])
            fp.write("{}\t{}\n".format(id_,status))

    with open(args.valid_ids_path,"w") as fp:
        for id_ in peptides.keys():
            fp.write("{}\n".format(id_))
        