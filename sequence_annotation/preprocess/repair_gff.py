import sys,os
from functools import cmp_to_key
import pandas as pd
pd.options.mode.chained_assignment = 'raise'
import numpy as np
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import get_gff_with_attribute,read_gff
from sequence_annotation.utils.utils import write_gff,GFF_COLUMNS, dupliacte_gff_by_parent
from sequence_annotation.preprocess.utils import GENE_TYPES,RNA_TYPES,EXON_TYPES,SUBEXON_TYPES
from argparse import ArgumentParser

def get_coord_id(block):
    id_ = None
    for column in GFF_COLUMNS[:5] + ['strand']:
        if id_ is None:
            id_ = str(block[column])
        else:
            id_ = id_ + "_" + str(block[column])
    return id_

def create_missing_UTRs(exons,subexons):
    missing_UTRs = []
    orf_start = orf_end = None
    CDSs_ = [e for e in subexons if e['feature'] == 'CDS']
    #If they beloings coding transcript
    if len(CDSs_) != 0:
        orf_start = min([CDS['start'] for CDS in CDSs_])
        orf_end = max([CDS['end'] for CDS in CDSs_])
        for exon in exons:
            #Create missing UTR
            strand = exon['strand']
            start = end = None
            selected_subexons = []
            for subexon in subexons:
                if not(subexon['end'] < exon['start'] or exon['end'] < subexon['start']):
                    selected_subexons.append(subexon)
            CDSs = [e for e in selected_subexons if e['feature'] == 'CDS']
            five_prime_UTRs = [e for e in selected_subexons if e['feature'] == 'five_prime_UTR']
            three_prime_UTRs = [e for e in selected_subexons if e['feature'] == 'three_prime_UTR']
            if len(CDSs) > 1:
                raise Exception("Wrong number of CDSs")
            if len(five_prime_UTRs) > 1:
                raise Exception("Wrong number of five_prime_UTRs")
            if len(three_prime_UTRs) > 1:
                raise Exception("Wrong number of three_prime_UTRs")

            if len(five_prime_UTRs)!=len(five_prime_UTRs):
                raise Exception("Inconsist annotation")
                
            #if there is no CDS in this exon, then make this exon into UTR
            if len(CDSs)==0:
                UTR = dict(exon)
                #Try to fix exon's 5'UTR or exon's 3'UTR, if exon dosn't have one
                if len(five_prime_UTRs+three_prime_UTRs)==0:
                    if strand=='+':
                        if exon['end'] < orf_start:
                            UTR['feature'] = 'five_prime_UTR'
                            missing_UTRs.append(UTR)
                        elif exon['start'] > orf_end:
                            UTR['feature'] = 'three_prime_UTR'
                            missing_UTRs.append(three_prime_UTR)
                    else:
                        if exon['start'] > orf_end:
                            UTR['feature'] = 'five_prime_UTR'
                            missing_UTRs.append(UTR)
                        elif exon['end'] < orf_start:
                            UTR['feature'] = 'three_prime_UTR'
                            missing_UTRs.append(three_prime_UTR)
            #if there is one CDS in this exon, then make create UTR by exon's ORF if it doesn't have one
            else:
                CDS = CDSs[0]
                if len(five_prime_UTRs)==0:
                    five_prime_UTR = dict(exon)
                    five_prime_UTR['feature'] = 'five_prime_UTR'
                    if strand == '+': 
                        five_prime_UTR['end'] = CDS['start'] - 1
                    else:
                        five_prime_UTR['start'] = CDS['end'] + 1
                    five_length = five_prime_UTR['end'] - five_prime_UTR['start'] + 1
                    if five_length > 0:
                        missing_UTRs.append(five_prime_UTR)

                if len(three_prime_UTRs)==0:
                    three_prime_UTR = dict(exon)
                    three_prime_UTR['feature'] = 'three_prime_UTR'
                    if strand == '+': 
                        three_prime_UTR['start'] = CDS['end'] + 1
                    else:   
                        three_prime_UTR['end'] = CDS['start'] - 1
                    three_length = three_prime_UTR['end'] - three_prime_UTR['start'] + 1
                    if three_length > 0:
                        missing_UTRs.append(three_prime_UTR)

    else:
        #Try to fix exon's UTR, if exon dosn't have one
        for exon in exons:
            UTR = dict(exon)
            UTR['feature'] = 'UTR'
            missing_UTRs.append(UTR)

    return missing_UTRs

def create_blocks(subblock_group,feature):
    """Create a list of block, if two blocks are neighbor, then they would be merged to one block"""
    starts = [item['start'] for item in subblock_group]
    ends = [item['end'] for item in subblock_group]
    template = dict(subblock_group[0])
    template['feature'] = feature
    del template['end']
    template['id'] = template['name'] = template['frame'] = '.'
    blocks_info = []
    indice = np.argsort(starts)
    temp = dict(template)
    previous_end = None
    for index in indice:
        start = starts[index]
        end = ends[index]
        if previous_end is None:
            temp['start'] = start
            temp['end'] = end
        else:
            if (previous_end+1) == start:
                temp['end'] = end
            else:
                blocks_info.append(temp)
                temp = dict(template)
                temp['start'] = start
                temp['end'] = end
        previous_end = end
    if 'end' in temp.keys():
        blocks_info.append(temp)
    return blocks_info

def create_exons(subblock_group):
    exons_info = create_blocks(subblock_group,'exon')
    return exons_info

def repair_by_subgroup(data,subdata):
    data_ = dict(data)
    if len(subdata) != 0:
        start = min([item['start'] for item in subdata])
        end = max([item['end'] for item in subdata])
        data_['start'] = start
        data_['end'] = end
    return data_

def repair_iterative(data,level,groups):
    #repair subtype data if exist
    repaired_subnodes = []
    subnode_subitems = []
    try:
        if level == 0:
            subitem_list = groups[0].get_group(data['id']).to_dict('record')
            repaired_subnodes = subitem_list
        else:
            selected = groups[level-1].get_group(data['id'])
            for subnodes in selected.to_dict('record'):
                repaired = repair_iterative(subnodes,level-1,groups)
                repaired_subnodes.append(repaired)
                subnode_subitems.append(repaired['node'])
    except KeyError:
        pass
    #repair data
    repaired_data = repair_by_subgroup(data,subnode_subitems)
    return {'node':repaired_data,'children':repaired_subnodes}

def add_iterative(data,returned):
    if 'node' in data.keys():
        returned.append(data['node'])
        for child in data['children']:
            add_iterative(child,returned)
    else:
        returned.append(data)

def gff_repair(gff):
    strand = set(gff['strand'])
    if len(strand - set(['+','-']))!=0:
        raise Exception("Wrong strand",strand)
    
    genes = gff[gff['feature'].isin(GENE_TYPES)]
    rnas = gff[gff['feature'].isin(RNA_TYPES)]
    exons = gff[gff['feature'].isin(EXON_TYPES)]
    subexons = gff[gff['feature'].isin(SUBEXON_TYPES)]
    
    exons = exons.assign(attribute = "ID=" + exons['id'] + ";Parent=" + exons['parent'] + ";Name=" + exons['name'])
    rna_group = rnas.groupby('parent')
    exon_group = exons.groupby('parent')
    subexon_group = subexons.groupby('parent')
    #Recreate subexon data
    exon_list = []
    subexons_list = []
    print("Create exon and UTR")
    index=0
    for rna_id in rnas['id']:
        print("Processed {}%".format(int(100*index/len(rnas))),end='\r')
        index+=1
        sys.stdout.write('\033[K')
        rna = rnas[rnas['id']==rna_id].to_dict('record')[0]
        try:
            exons_ = exon_group.get_group(rna_id).to_dict('record')
        except KeyError:
            continue
        #If subexon exists, then try to use it to repair UTRs and exons and exons to list
        #Otherwise, exons to list
        try:
            subexons_ = subexon_group.get_group(rna_id).to_dict('record')
            missing_UTRs_ = create_missing_UTRs(exons_,subexons_)
            subexons_ += missing_UTRs_
            created_subexons = []
            for type_ in SUBEXON_TYPES:
                list_ = []
                for subexon in subexons_:
                    if subexon['feature'] == type_:
                        list_.append(subexon)
                if len(list_) > 0:
                    created_subexons += create_blocks(list_,type_)
            subexons_list += created_subexons
            created_exons = create_exons(created_subexons)
            if len(exons_) != len(created_exons):
                raise Exception("Inonsist exon number at {}, got {} and {}".format(rna_id,len(exons_),
                                                                                   len(created_exons)))
            exon_list += created_exons
            
        except KeyError:
            exon_list += exons_

    all_exons = pd.DataFrame.from_dict(subexons_list + exon_list)
    all_exons.loc[:,'attribute'] = "ID=" + all_exons['id'] + ";Parent=" + all_exons['parent'] + ";Name=" + all_exons['name']
    all_exons_group = all_exons.groupby('parent')

    gene_node = []
    groups = [all_exons_group,rna_group]
    index=0
    print("Repair data")
    for gene in genes.to_dict('record'):
        print("Processed {}%".format(int(100*index/len(genes))),end='\r')
        index+=1
        sys.stdout.write('\033[K')
        node = repair_iterative(gene,len(groups),groups)
        gene_node.append(node)

    returned = []
    for gene in gene_node:
        add_iterative(gene,returned)
        
    all_types = GENE_TYPES + RNA_TYPES + EXON_TYPES + SUBEXON_TYPES    
    others = gff[~gff['feature'].isin(all_types)].to_dict('record') 
    returned += others

    returned = pd.DataFrame.from_dict(returned)
    return returned

if __name__ =='__main__':
    parser = ArgumentParser(description="This program will use CDS and exon to "+
                            "repair missing UTR, use subexon to repair exon\n"+
                            ", and use exon to repiar rna, and use rna to repair gene.")
    parser.add_argument("-i", "--input_path",help="Path of input GFF file",required=True)
    parser.add_argument("-o", "--output_path",help="Path of output GFFfile",required=True)
    parser.add_argument("-s", "--saved_root",help="Root to save broken and repaired item",required=True)
    args = parser.parse_args()

    gff = read_gff(args.input_path)
    gff = get_gff_with_attribute(gff,['parent'])    
    gff = dupliacte_gff_by_parent(gff)

    created_gff = gff_repair(gff)
    write_gff(created_gff,args.output_path)

    origin_ids = [get_coord_id(item) for item in gff.to_dict('record')]
    created_ids = [get_coord_id(item) for item in created_gff.to_dict('record')]
    gff.loc[:,'coord_id'] = origin_ids
    created_gff.loc[:,'coord_id'] = created_ids
    broken_ids = set(origin_ids) - set(created_ids)
    repaired_ids = set(created_ids) - set(origin_ids)
    
    broken_gff = gff[gff['coord_id'].isin(broken_ids)]
    repaired_gff = created_gff[created_gff['coord_id'].isin(repaired_ids)]
    write_gff(broken_gff,os.path.join(args.saved_root,'broken.gff'))
    write_gff(repaired_gff,os.path.join(args.saved_root,'repaired.gff'))
