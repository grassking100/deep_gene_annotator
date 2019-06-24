import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import classify_data_by_id, get_id_table, read_bed, write_bed
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    root_path = "/".join(sys.argv[0].split('/')[:-1])
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root",help="saved_root",required=True)
    parser.add_argument("-i", "--id_convert_path",help="id_convert_path",required=False)
    parser.add_argument("-r", "--raw_bed_path",help="raw_bed_path",required=True)
    parser.add_argument("-c", "--coordinate_consist_bed_path",help="coordinate_consist_bed_path",required=True)
    parser.add_argument("-f", "--fai_path",help="fai_path",required=True)
    parser.add_argument("-u", "--upstream_dist",help="upstream_dist",type=int,required=True)
    parser.add_argument("-d", "--downstream_dist",help="downstream_dist",type=int,required=True)
    args = parser.parse_args()
    output_path = os.path.join(args.saved_root,"recurrent_cleaned.bed")
    if os.path.exists(output_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        ###Read data###
        id_convert = None
        if args.id_convert_path is not None:
            id_convert = get_id_table(args.id_convert_path)
        raw_bed = read_bed(args.raw_bed_path)
        coordinate_consist_bed = read_bed(args.coordinate_consist_bed_path)
        raw_bed['chr'] = raw_bed['chr'].str.replace('Chr', '')
        want_id = set(coordinate_consist_bed['id'])
        all_id = set(raw_bed['id'])
        left_bed = raw_bed[~raw_bed['id'].isin(want_id)]
        all_bed = pd.concat([coordinate_consist_bed,left_bed])
        want_bed_path = os.path.join(args.saved_root,'want.bed')
        unwant_bed_path = os.path.join(args.saved_root,'unwant.bed')
        saved_nums = []
        saved_num = -1
        index = 0
        ###Recurrent cleaning###
        while True:
            index += 1
            want_bed, unwant_bed = classify_data_by_id(all_bed,want_id,id_convert)
            write_bed(want_bed,want_bed_path)
            write_bed(unwant_bed,unwant_bed_path)
            if len(want_bed)==0:
                break
            path = 'region_upstream_{}_downstream_{}_safe_zone_id'.format(args.upstream_dist,args.downstream_dist)
            id_path = os.path.join(args.saved_root,path)
            command = "bash {}/../bash/safe_filter.sh {} {} {} {} {} {}"
            command = command.format(root_path,want_bed_path,unwant_bed_path,
                                     args.fai_path,args.upstream_dist,
                                     args.downstream_dist,args.saved_root)
            print("Execute command:"+command)
            os.system(command)
            #break
            want_id = [id_ for id_ in open(id_path+'.txt').read().split('\n') if id_ != '']
            with open('{}_{}.txt'.format(id_path,index),"w") as fp:
                for id_ in want_id:
                    fp.write(id_+"\n")
            num = len(want_id)
            print("Iter "+str(index)+" get number:"+str(num))
            saved_nums.append(num)
            if num == saved_num:
                break
            else:
                saved_num = num
                want_bed_path = os.path.join(args.saved_root,'want_iter_{}.bed'.format(index))
                unwant_bed_path = os.path.join(args.saved_root,'unwant_iter_{}.bed'.format(index))

        ###Write data###
        want_bed = coordinate_consist_bed[coordinate_consist_bed['id'].isin(want_id)]
        want_bed = want_bed[~ want_bed.duplicated()]
        write_bed(want_bed,output_path)
        with open(os.path.join(args.saved_root,'recurrent_count.stats'),'w') as fp:
            for index,num in  enumerate(saved_nums):
                fp.write("Iterate "+str(index+1)+":left "+str(num)+" transcript.\n")
        