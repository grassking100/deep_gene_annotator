import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import read_bed, write_bed, simply_coord
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root",help="saved_root",required=True)
    parser.add_argument("-b", "--bed_path",help="bed_path",required=True)
    parser.add_argument("-g", "--genome_path",help="genome_path",required=True)
    parser.add_argument("-t", "--TSS_radius",help="TSS_radius",required=True)
    parser.add_argument("-c", "--cleavage_radius",help="cleavage_radius",required=True)
    parser.add_argument("-d", "--donor_radius",help="donor_radius",required=True)
    parser.add_argument("-a", "--accept_radius",help="accept_radius",required=True)
    args = vars(parser.parse_args())
    saved_root = args['saved_root']
    bed_path = args['bed_path']
    genome_path = args['genome_path']
    TSS_radius = args['TSS_radius']
    cleavage_radius = args['cleavage_radius']
    donor_radius = args['donor_radius']
    accept_radius = args['accept_radius']
    root_path = "/".join(sys.argv[0].split('/')[:-1])
    script_root = root_path+'/../bash'
    tss_around_path = 'transcription_start_site_with_radius_'+str(TSS_radius)
    cleavage_around_path = 'cleavage_site_with_radius_'+str(cleavage_radius)
    donor_around_path = 'splice_donor_site_with_radius_'+str(donor_radius)
    accept_around_path = 'splice_accept_site_with_radius_'+str(accept_radius)
    for tool_name,dist in zip(['transcription_start_site','splice_donor_site','splice_accept_site','cleavage_site'],
                              [TSS_radius,donor_radius,accept_radius,cleavage_radius]):
        command = "bash "+script_root+"/"+tool_name+".sh "+bed_path+" "+str(dist)
        print(command)
        os.system(command)
    paths = [tss_around_path,cleavage_around_path,donor_around_path,accept_around_path]
    for path in paths:
        path = saved_root+"/result_"+path+'.bed'
        write_bed(simply_coord(read_bed(path)),path)
    for path_root in paths:
        command = 'bedtools getfasta -s -fi '+genome_path+' -bed '+saved_root+"/result_"
        command += path_root+'.bed -fo '+saved_root+"/result_"+ path_root+'.fasta'
        print(command)
        os.system(command)
