import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import read_bed, write_bed, simply_coord, merge_bed_by_coord
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root",help="saved_root",required=True)
    parser.add_argument("-b", "--bed_path",help="bed_path",required=True)
    parser.add_argument("-g", "--genome_path",help="genome_path",required=True)
    parser.add_argument("-u", "--upstream_dist",help="upstream_dist",required=True)
    parser.add_argument("-o", "--downstream_dist",help="downstream_dist",required=True)
    parser.add_argument("-t", "--TSS_radius",help="TSS_radius",required=True)
    parser.add_argument("-c", "--cleavage_radius",help="cleavage_radius",required=True)
    parser.add_argument("-d", "--donor_radius",help="donor_radius",required=True)
    parser.add_argument("-a", "--accept_radius",help="accept_radius",required=True)
    args = vars(parser.parse_args())
    saved_root = args['saved_root']
    bed_path = args['bed_path']
    genome_path = args['genome_path']
    upstream_dist = args['upstream_dist']
    downstream_dist = args['downstream_dist']
    TSS_radius = args['TSS_radius']
    cleavage_radius = args['cleavage_radius']
    donor_radius = args['donor_radius']
    accept_radius = args['accept_radius']
    script_root = './sequence_annotation/gene_info/script/bash'
    result_bed_path = "result"
    around_path = 'upstream_'+str(upstream_dist)+'_downstream_'+str(downstream_dist)
    tss_around_path = 'transcription_start_site_with_radius_'+str(TSS_radius)
    cleavage_around_path = 'cleavage_site_with_radius_'+str(cleavage_radius)
    donor_around_path = 'splice_donor_site_with_radius_'+str(donor_radius)
    accept_around_path = 'splice_accept_site_with_radius_'+str(accept_radius)
    for tool_name,dist in zip(['transcription_start_site','splice_donor_site','splice_accept_site','cleavage_site'],
                              [TSS_radius,donor_radius,accept_radius,cleavage_radius]):
        command = "bash "+script_root+"/"+tool_name+".sh "+saved_root+"/"+result_bed_path+".bed "+str(dist)
        print(command)
        os.system(command)
    command = "bash "+script_root+"/selected_around.sh "+saved_root+"/"+result_bed_path+".bed "
    command += str(upstream_dist)+" "+str(downstream_dist)
    print(command)
    os.system(command)
    paths = [tss_around_path,cleavage_around_path,donor_around_path,accept_around_path,around_path]
    for path in paths:
        path = saved_root+"/"+result_bed_path+"_"+path+'.bed'
        write_bed(simply_coord(read_bed(path)),path)
    result_bed = read_bed(saved_root+"/"+result_bed_path+".bed")
    result_bed = merge_bed_by_coord(result_bed)
    id_ = result_bed['id'].astype(str)
    id_ += '_around_upstream_'+str(upstream_dist)+'_downstream_'+str(downstream_dist)
    result_bed['id'] = id_
    merged_path = saved_root+"/"+result_bed_path+"_"+around_path+"_merged"
    write_bed(result_bed,merged_path+".bed")
    command = 'bedtools getfasta -s -name -fi '+genome_path+' -bed '+merged_path+'.bed -fo '+merged_path+'.fasta'
    os.system(command)
    for path_root in paths:
        command = 'bedtools getfasta -s -name -fi '+genome_path+' -bed '+saved_root+"/"
        command += path_root+'.bed -fo '+saved_root+"/"+ path_root+'.fasta'
        print(command)
        os.system(command)
