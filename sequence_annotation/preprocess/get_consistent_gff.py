import sys, os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_gff, write_gff, GFF_COLUMNS, get_gff_with_attribute, get_gff_with_updated_attribute
from sequence_annotation.preprocess.utils import GENE_TYPES, RNA_TYPES, EXON_TYPES, SUBEXON_TYPES
from argparse import ArgumentParser


def _get_coord_ids(gff):
    ids = []
    for block in gff.to_dict('record'):
        id_ = None
        for column in GFF_COLUMNS[:5] + ['strand']:
            if id_ is None:
                id_ = str(block[column])
            else:
                id_ = id_ + "_" + str(block[column])
        ids.append(id_)
    return ids


def _create_missing_UTRs(exons, subexons):
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
                if not (subexon['end'] < exon['start']
                        or exon['end'] < subexon['start']):
                    selected_subexons.append(subexon)
            CDSs = [e for e in selected_subexons if e['feature'] == 'CDS']
            five_prime_UTRs = [
                e for e in selected_subexons
                if e['feature'] == 'five_prime_UTR'
            ]
            three_prime_UTRs = [
                e for e in selected_subexons
                if e['feature'] == 'three_prime_UTR'
            ]
            if len(CDSs) > 1:
                raise Exception("Wrong number of CDSs")
            if len(five_prime_UTRs) > 1:
                raise Exception("Wrong number of five_prime_UTRs")
            if len(three_prime_UTRs) > 1:
                raise Exception("Wrong number of three_prime_UTRs")

            #If there is no CDS in this exon, then create UTR for this exon
            if len(CDSs) == 0:
                UTR = dict(exon)
                #Try to fix exon's 5'UTR or exon's 3'UTR, if exon doesn't have one
                if len(five_prime_UTRs + three_prime_UTRs) == 0:
                    if strand == '+':
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
            #If there is one CDS in this exon, then make create UTR by exon's ORF if it doesn't have one
            else:
                CDS = CDSs[0]
                if len(five_prime_UTRs) == 0:
                    five_prime_UTR = dict(exon)
                    five_prime_UTR['feature'] = 'five_prime_UTR'
                    if strand == '+':
                        five_prime_UTR['end'] = CDS['start'] - 1
                    else:
                        five_prime_UTR['start'] = CDS['end'] + 1
                    five_length = five_prime_UTR['end'] - five_prime_UTR[
                        'start'] + 1
                    if five_length > 0:
                        missing_UTRs.append(five_prime_UTR)

                if len(three_prime_UTRs) == 0:
                    three_prime_UTR = dict(exon)
                    three_prime_UTR['feature'] = 'three_prime_UTR'
                    if strand == '+':
                        three_prime_UTR['start'] = CDS['end'] + 1
                    else:
                        three_prime_UTR['end'] = CDS['start'] - 1
                    three_length = three_prime_UTR['end'] - three_prime_UTR[
                        'start'] + 1
                    if three_length > 0:
                        missing_UTRs.append(three_prime_UTR)

    else:
        #Try to create exon's UTR, if exon doesn't have one
        for exon in exons:
            UTR = dict(exon)
            UTR['feature'] = 'UTR'
            missing_UTRs.append(UTR)

    return missing_UTRs


def _create_blocks(subblock_list, feature):
    """Create a list of block, if two blocks are neighbor, then they would be merged to one block"""
    starts = [item['start'] for item in subblock_list]
    ends = [item['end'] for item in subblock_list]
    template = dict(subblock_list[0])
    template['feature'] = feature
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
            if (previous_end + 1) == start:
                temp['end'] = end
            else:
                blocks_info.append(temp)
                temp = dict(template)
                temp['start'] = start
                temp['end'] = end
        previous_end = end
    blocks_info.append(temp)
    return blocks_info


def _create_exons(subblock_list):
    exons_info = _create_blocks(subblock_list, 'exon')
    return exons_info


def _get_item_with_repaired_boundary(parent, child_list):
    parent = dict(parent)
    if len(child_list) != 0:
        start = min([item['start'] for item in child_list])
        end = max([item['end'] for item in child_list])
        parent['start'] = start
        parent['end'] = end
    return parent


def _get_repaired_subexon_and_exon(transcript_id, exon_group, subexon_group):
    #If subexon exists, then try to use it to repair UTRs and exons and add reapired exons to list
    #Otherwise, directly add exons to list
    exon_list = []
    subexons_list = []
    if transcript_id in list(exon_group.groups.keys()):
        exons = exon_group.get_group(transcript_id).to_dict('record')
        if transcript_id in list(subexon_group.groups.keys()):
            subexons = subexon_group.get_group(transcript_id).to_dict('record')
            missing_UTRs_ = _create_missing_UTRs(exons, subexons)
            subexons += missing_UTRs_
            created_subexons = []
            for type_ in SUBEXON_TYPES:
                list_ = [
                    subexon for subexon in subexons
                    if subexon['feature'] == type_
                ]
                if len(list_) > 0:
                    created_subexons += _create_blocks(list_, type_)
            subexons_list += created_subexons
            created_exons = _create_exons(created_subexons)
            if len(exons) != len(created_exons):
                raise Exception(
                    "Inconsist exon number at {}, got {} and {}".format(
                        transcript_id, len(exons), len(created_exons)))
            exon_list += created_exons
        else:
            exon_list += exons
    return subexons_list, exon_list


def _group_list_by_parent(item_list):
    group = {}
    for item in item_list:
        parent = item['parent']
        if parent not in group:
            group[parent] = []
        group[parent].append(item)
    return group


def repair_gff(gff):
    strand = set(gff['strand'])
    if len(strand - set(['+', '-'])) != 0:
        raise Exception("Wrong strand", strand)

    genes = gff[gff['feature'].isin(GENE_TYPES)]
    transcripts = gff[gff['feature'].isin(RNA_TYPES)]
    exons = gff[gff['feature'].isin(EXON_TYPES)]
    subexons = gff[gff['feature'].isin(SUBEXON_TYPES)]
    exon_group = exons.groupby('parent')
    subexon_group = subexons.groupby('parent')
    transcript_group = transcripts.groupby('parent')
    #Recreate subexon data
    gene_list = genes.to_dict('record')
    transcript_list = transcripts.to_dict('record')
    exon_list = []
    subexons_list = []
    print("Create exon and UTR")
    for index, transcript_id in enumerate(list(transcripts['id'])):
        print("Processed {}%".format(int(100 * index / len(transcripts))),
              end='\r')
        sys.stdout.write('\033[K')
        subexons_list_, exon_list_ = _get_repaired_subexon_and_exon(
            transcript_id, exon_group, subexon_group)
        exon_list += exon_list_
        subexons_list += subexons_list_

    repaired = exon_list + subexons_list
    repaired_transcripts = []
    exon_list_group = _group_list_by_parent(exon_list)

    print("Repair transcript boundary if transcript has any exon")
    for index, transcript in enumerate(transcript_list):
        print("Processed {}%".format(int(100 * index / len(transcript_list))),
              end='\r')
        sys.stdout.write('\033[K')
        transcript_id = transcript['id']
        if transcript_id in exon_list_group:
            matched_list = exon_list_group[transcript_id]
            transcript = _get_item_with_repaired_boundary(
                transcript, matched_list)
        repaired_transcripts.append(transcript)
        repaired.append(transcript)

    print("Repair gene boundary")
    for index, gene in enumerate(gene_list):
        print("Processed {}%".format(int(100 * index / len(genes))), end='\r')
        sys.stdout.write('\033[K')
        matched_list = transcript_group.get_group(gene['id']).to_dict('record')
        gene = _get_item_with_repaired_boundary(gene, matched_list)
        repaired.append(gene)

    ALL_types = GENE_TYPES + RNA_TYPES + EXON_TYPES + SUBEXON_TYPES
    others = gff[~gff['feature'].isin(ALL_types)].to_dict('record')
    repaired += others
    repaired = pd.DataFrame.from_dict(repaired)
    repaired.loc[:, 'coord_id'] = _get_coord_ids(repaired)
    repaired = get_gff_with_updated_attribute(repaired)
    return repaired


def main(input_path, saved_root, postfix):
    gff = read_gff(input_path)
    gff = get_gff_with_attribute(gff)
    gff.loc[:, 'coord_id'] = _get_coord_ids(gff)

    repaired_gff = repair_gff(gff)
    write_gff(repaired_gff,
              os.path.join(saved_root, 'repaired_{}.gff3').format(postfix))

    origin_coord_ids = set(gff['coord_id'])
    created_coord_ids = set(repaired_gff['coord_id'])

    broken_ids = origin_coord_ids - created_coord_ids
    unseen_ids = created_coord_ids - origin_coord_ids

    broken_block_gff = gff[gff['coord_id'].isin(broken_ids)]
    unseen_block_gff = repaired_gff[repaired_gff['coord_id'].isin(unseen_ids)]

    broken_gene_ids = set(broken_block_gff['belong_gene'])
    unseen_gene_ids = set(unseen_block_gff['belong_gene'])
    inconsistent_gene_id = unseen_gene_ids.union(broken_gene_ids)
    inconsistent_gff = repaired_gff[repaired_gff['belong_gene'].isin(
        inconsistent_gene_id)]
    consistent_gff = repaired_gff[~repaired_gff['belong_gene'].
                                  isin(inconsistent_gene_id)]

    write_gff(inconsistent_gff,
              os.path.join(saved_root, 'inconsistent_{}.gff3').format(postfix))
    write_gff(consistent_gff,
              os.path.join(saved_root, 'consistent_{}.gff3').format(postfix))
    write_gff(broken_block_gff,
              os.path.join(saved_root, 'broken_block_{}.gff3').format(postfix))
    write_gff(unseen_block_gff,
              os.path.join(saved_root, 'unseen_block_{}.gff3').format(postfix))


if __name__ == '__main__':
    parser = ArgumentParser(
        description="This program would use CDS and exon to " +
        "repair missing UTR, use CDS and UTR to repair exon\n" +
        ", and use exon to repair transcript, and use transcript to repair gene."
        + "Output data would be consistent data which are not changed.")
    parser.add_argument("-i",
                        "--input_path",
                        help="Path of input GFF file",
                        required=True)
    parser.add_argument("-s",
                        "--saved_root",
                        help="Root to save result",
                        required=True)
    parser.add_argument("-p", "--postfix", required=True)
    args = parser.parse_args()

    kwargs = vars(args)

    main(**kwargs)
