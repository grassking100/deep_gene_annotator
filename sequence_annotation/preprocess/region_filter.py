import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed, write_bed
from sequence_annotation.genome_handler.exception import InvalidStrandType

def region_filter(region_bed,gene_bed,upstream_distance,downstream_distance):
    regions = region_bed.to_dict('record')
    invalid_strand = gene_bed[~gene_bed['strand'].isin(['+','-'])]
    if len(invalid_strand) != 0:
        raise InvalidStrandType("Invalid strand exist in RNA data")

    valid_bed_items = []
    invalid_bed_items = []
    
    for region in regions:
        gene_ids = set(region['id'].split(","))
        if "." in gene_ids:
            gene_ids.remove(".")
        selected = gene_bed[gene_bed['id'].isin(gene_ids)]
        has_partial = False
        for gene in selected.to_dict('record'):
            if gene['strand'] == '+':
                start = gene['start']-upstream_distance
                end = gene['end']+downstream_distance
            else:   
                start = gene['start']-downstream_distance
                end = gene['end']+upstream_distance

            if not (region['start'] <= start and end <= region['end']):
                has_partial = True

        if has_partial:
            invalid_bed_items.append(region)
        else:    
            count = {'+':0,'-':0}
            for gene_id in gene_ids:
                subset = selected[selected['id'] == gene_id]
                strands = set(list(subset['strand']))
                if len(strands)==1:
                    count[list(strands)[0]] += 1
                else:
                    raise Exception("Inconsist strand for same gene, {}".format(gene_id))
            if count['+'] <= 1 and count['-'] <= 1:
                valid_bed_items.append(region)
            else:
                invalid_bed_items.append(region)
    valid_bed = pd.DataFrame.from_dict(valid_bed_items)
    invalid_bed = pd.DataFrame.from_dict(invalid_bed_items)
    return valid_bed, invalid_bed

def main(region_bed_path,gene_bed_path,region_output_path,upstream_distance,downstream_distance,discard_output_path=None):
    region_bed = read_bed(region_bed_path)
    gene_bed = read_bed(gene_bed_path)
    valid_bed, invalid_bed = region_filter(region_bed,gene_bed,upstream_distance,downstream_distance)
    write_bed(valid_bed,region_output_path)
    if discard_output_path is not None:
        write_bed(invalid_bed,discard_output_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="This program will print region which have at most one gene"+
                            "(with specific distance) at both strands, it will also discard"+
                            "regions contain partial gene")
    parser.add_argument("-i", "--region_bed_path",required=True,
                        help='Regions which have at most one gene at each strand')
    parser.add_argument("-r", "--gene_bed_path",required=True)
    parser.add_argument("-u", "--upstream_distance",type=int,required=True)
    parser.add_argument("-d", "--downstream_distance",type=int,required=True)
    parser.add_argument("-o", "--region_output_path",required=True)
    parser.add_argument("-e", "--discard_output_path")
    args = parser.parse_args()
    main(args.region_bed_path,args.gene_bed_path,args.region_output_path,
         args.upstream_distance,args.downstream_distance,
         args.discard_output_path)

