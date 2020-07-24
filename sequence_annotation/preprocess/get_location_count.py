import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.preprocess.consist_site import preprocess, get_consist_site

def main(external_UTR_path,transcript_path,long_dist_path,output_path,
         remove_transcript_external_UTR_conflict):
        ###Read file###
        on_external_UTR = preprocess(external_UTR_path)
        on_ld = preprocess(long_dist_path)
        on_transcript = preprocess(transcript_path)
        data = get_consist_site(on_external_UTR, on_ld, on_transcript,
                                remove_conflict=remove_transcript_external_UTR_conflict)
        data[['coord_id','feature']].drop_duplicates()['feature'].value_counts().to_csv(output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-u","--external_UTR_path", required=True)
    parser.add_argument("-l","--long_dist_path", required=True)
    parser.add_argument("-t","--transcript_path", required=True)
    parser.add_argument("-o","--output_path", required=True)
    parser.add_argument("--remove_transcript_external_UTR_conflict",
                        action='store_true')
    
    args = parser.parse_args()
    main(**vars(args))
