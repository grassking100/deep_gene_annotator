import sys, os
import numpy as np
import pandas as pd
from multiprocessing import Pool
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import read_gff, write_gff, GFF_COLUMNS
from sequence_annotation.file_process.utils import get_gff_with_attribute, get_gff_with_updated_attribute
from sequence_annotation.file_process.utils import GENE_TYPE, TRANSCRIPT_TYPE, EXON_TYPE
from sequence_annotation.file_process.utils import CDS_TYPE, SUBEXON_TYPES
from sequence_annotation.file_process.utils import UTR_TYPE

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


def _create_missing_UTRs(subexons,exons):
    missing_UTRs = []
    orf_start = orf_end = None
    coding_seqs = [e for e in subexons if e['feature']==CDS_TYPE]
    #If they beloings coding transcript
    if len(coding_seqs) != 0:
        orf_start = min([coding_seq['start'] for coding_seq in coding_seqs])
        orf_end = max([coding_seq['end'] for coding_seq in coding_seqs])
        for exon in exons:
            #Create missing UTR
            strand = exon['strand']
            #start = end = None
            selected_subexons = []
            for subexon in subexons:
                if not (subexon['end'] < exon['start'] or exon['end'] < subexon['start']):
                    selected_subexons.append(subexon)
            coding_seqs = [e for e in selected_subexons if e['feature']==CDS_TYPE]
            five_prime_UTRs = []
            three_prime_UTRs = []
            if strand == '+':
                for e in selected_subexons:
                    if e['end'] < orf_start:
                        five_prime_UTRs.append(e)
                    if orf_end < e['start']:
                        three_prime_UTRs.append(e)
            else:
                for e in selected_subexons:
                    if e['end'] < orf_start:
                        three_prime_UTRs.append(e)
                    if orf_end < e['start']:
                        five_prime_UTRs.append(e)
            if len(coding_seqs) > 1:
                raise Exception("Wrong number of coding_seqs")
            if len(five_prime_UTRs) > 1:
                raise Exception("Wrong number of five_prime_UTRs")
            if len(three_prime_UTRs) > 1:
                raise Exception("Wrong number of three_prime_UTRs")
            #If there is no coding_seq in this exon, then create UTR for this exon
            if len(coding_seqs) == 0:
                UTR = dict(exon)
                UTR['feature'] = UTR_TYPE
                #Try to fix exon's 5'UTR or exon's 3'UTR, if exon doesn't have one
                if len(five_prime_UTRs + three_prime_UTRs) == 0:
                    if strand == '+':
                        if exon['end'] < orf_start:
                            #UTR['feature'] = FIVE_PRIME_UTR_TYPE
                            missing_UTRs.append(UTR)
                        elif exon['start'] > orf_end:
                            #UTR['feature'] = THREE_PRIME_UTR_TYPE
                            missing_UTRs.append(UTR)
                    else:
                        if exon['start'] > orf_end:
                            #UTR['feature'] = FIVE_PRIME_UTR_TYPE
                            missing_UTRs.append(UTR)
                        elif exon['end'] < orf_start:
                            #UTR['feature'] = THREE_PRIME_UTR_TYPE
                            missing_UTRs.append(UTR)
            #If there is one coding_seq in this exon, then make create UTR by exon's ORF if it doesn't have one
            else:
                coding_seq = coding_seqs[0]
                if len(five_prime_UTRs) == 0:
                    UTR = dict(exon)
                    UTR['feature'] = UTR_TYPE
                    #UTR['feature'] = FIVE_PRIME_UTR_TYPE
                    if strand == '+':
                        UTR['end'] = coding_seq['start'] - 1
                    else:
                        UTR['start'] = coding_seq['end'] + 1
                    five_length = UTR['end'] - UTR['start'] + 1
                    if five_length > 0:
                        missing_UTRs.append(UTR)

                if len(three_prime_UTRs) == 0:
                    UTR = dict(exon)
                    UTR['feature'] = UTR_TYPE
                    #UTR['feature'] = THREE_PRIME_UTR_TYPE
                    if strand == '+':
                        UTR['start'] = coding_seq['end'] + 1
                    else:
                        UTR['end'] = coding_seq['start'] - 1
                    three_length = UTR['end'] - UTR['start'] + 1
                    if three_length > 0:
                        missing_UTRs.append(UTR)

    else:
        #Try to create exon's UTR, if exon doesn't have coding_seq
        for exon in exons:
            UTR = dict(exon)
            UTR['feature'] = UTR_TYPE
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


def _get_item_with_repaired_boundary(parent, child_list):
    parent = dict(parent)
    if len(child_list) > 0:
        start = min([item['start'] for item in child_list])
        end = max([item['end'] for item in child_list])
        parent['start'] = start
        parent['end'] = end
    return parent


def _grouping(blocks,by=None):
    by = by or 'parent'
    dict_ = {}
    for block in blocks:
        parent = block[by]
        if parent not in dict_:
            dict_[parent] = []
        dict_[parent].append(block)
    return dict_


def _repaired_subexon_and_exon(transcript_id, subexons, exons):
    print("Repair subexon and exon for {}".format(transcript_id))
    #If subexon exists, then try to use it to repair UTRs and exons and add reapired exons to list
    #Otherwise, directly add exons to list
    if len(subexons) > 0:
        created_subexons = []
        missing_UTRs = _create_missing_UTRs(subexons,exons)
        subexons += missing_UTRs
        subexons_dict = _grouping(subexons,'feature')
        for feature,subexons_ in subexons_dict.items():
            created_subexons += _create_blocks(subexons_, feature)
        created_exons = _create_blocks(created_subexons, 'exon')
        if len(exons) != len(created_exons):
            raise Exception("Inconsist exon number at {}, got {} and {}".format(transcript_id, 
                                                                                len(exons), 
                                                                                len(created_exons)))
        subexons = created_subexons
        exons = created_exons
        
    return transcript_id,subexons,exons


def _repair_transcript(exon_list,transcript):
    transcript_id = transcript['id']
    print("Repair transcript for {}".format(transcript_id))
    transcript = _get_item_with_repaired_boundary(transcript, exon_list)
    return transcript


def _repair_gene(transcript_list,gene):
    print("Repair gene for {}".format(gene['id']))
    gene = _get_item_with_repaired_boundary(gene, transcript_list)
    return gene


def repair_subexon_exon(subexons,exons):
    exon_dict = _grouping(exons.to_dict('record'))
    subexon_dict = _grouping(subexons.to_dict('record'))
    repaired_exon_dict = {}
    repaired_subexon_dict = {}
    args_list = []
    transcript_ids = list(set(exons['parent']))
    for index, transcript_id in enumerate(transcript_ids):
        print("Processed {}%".format(int(100 * index / len(transcript_ids))),end='\r')
        sys.stdout.write('\033[K')
        subexons_ = []
        exons_ = exon_dict[transcript_id]
        if transcript_id in subexon_dict:
            subexons_ = subexon_dict[transcript_id]
        args_list.append((transcript_id, subexons_, exons_))
        
    with Pool(processes=40) as pool:
        result_tuples = pool.starmap(_repaired_subexon_and_exon,args_list)
    
    for transcript_id, subexons_list, exon_list in result_tuples:
        repaired_subexon_dict[transcript_id] = subexons_list
        repaired_exon_dict[transcript_id] = exon_list
    return repaired_subexon_dict,repaired_exon_dict


def repair_transcript(repaired_exon_dict,transcripts):
    repaired_transcript_dict = {}
    repaired_args_list = []
    for index, transcript in enumerate(transcripts.to_dict('record')):
        print("Processed {}%".format(int(100 * index / len(transcripts))),end='\r')
        sys.stdout.write('\033[K')
        id_ = transcript['id']
        exons = []
        if id_ in repaired_exon_dict:
            exons = repaired_exon_dict[id_]
        repaired_args_list.append((exons,transcript))

    with Pool(processes=40) as pool:
        results = pool.starmap(_repair_transcript,repaired_args_list)
    repaired_transcript_dict = _grouping(results)
    return repaired_transcript_dict


def repair_gene(repaired_transcript_dict,genes):
    repaired_gene_dict = {}
    repaired_args_list = []
    for index, gene in enumerate(genes.to_dict('records')):
        print("Processed {}%".format(int(100 * index / len(genes))),end='\r')
        sys.stdout.write('\033[K')
        id_ = gene['id']
        if id_ in repaired_transcript_dict:
            transcripts = repaired_transcript_dict[id_]
        repaired_args_list.append((transcripts,gene))

    with Pool(processes=40) as pool:
        results = pool.starmap(_repair_gene,repaired_args_list)
    for gene in results:
        repaired_gene_dict[gene['id']] = gene
    return repaired_gene_dict


def repair_gff(gff):
    gff = gff.copy()
    strand = set(gff['strand'])
    invalid_strands = strand - set(['+', '-'])
    if len(invalid_strands) > 0:
        raise Exception("Wrong strand", invalid_strands)

    genes = gff[gff['feature']==GENE_TYPE]
    transcripts = gff[gff['feature']==TRANSCRIPT_TYPE]
    exons = gff[gff['feature']==EXON_TYPE]
    subexons = gff[gff['feature'].isin(SUBEXON_TYPES)]
    print("Create exon and UTR")
    repaired_subexon_dict,repaired_exon_dict = repair_subexon_exon(subexons,exons)
    print("Repair transcript boundary if transcript has any exon")
    repaired_transcript_dict = repair_transcript(repaired_exon_dict,transcripts)
    print("Repair gene boundary")
    repaired_gene_dict = repair_gene(repaired_transcript_dict,genes)
    repaired = []
    for list_ in list(repaired_subexon_dict.values()):
        repaired += list_
    for list_ in list(repaired_exon_dict.values()):
        repaired += list_
    for list_ in list(repaired_transcript_dict.values()):
        repaired += list_
    repaired += list(repaired_gene_dict.values())

    all_types = [GENE_TYPE,TRANSCRIPT_TYPE,EXON_TYPE] + SUBEXON_TYPES
    repaired = pd.DataFrame.from_dict(repaired)
    repaired = pd.concat([repaired,gff[~gff['feature'].isin(all_types)]])
    repaired.loc[:, 'coord_id'] = _get_coord_ids(repaired)
    repaired = get_gff_with_updated_attribute(repaired)
    return repaired


def main(input_path, saved_root, postfix):
    gff = read_gff(input_path)
    gff = get_gff_with_attribute(gff)
    gff.loc[:, 'coord_id'] = _get_coord_ids(gff)
    repaired_gff = repair_gff(gff)
    repaired_path = os.path.join(saved_root,'repaired_{}.gff3').format(postfix)
    write_gff(repaired_gff,repaired_path)

    origin_coord_ids = set(gff['coord_id'])
    created_coord_ids = set(repaired_gff['coord_id'])

    broken_ids = origin_coord_ids - created_coord_ids
    unseen_ids = created_coord_ids - origin_coord_ids

    broken_block_gff = gff[gff['coord_id'].isin(broken_ids)]
    unseen_block_gff = repaired_gff[repaired_gff['coord_id'].isin(unseen_ids)]

    broken_gene_ids = set(broken_block_gff['belong_gene'])
    unseen_gene_ids = set(unseen_block_gff['belong_gene'])
    inconsistent_gene_id = unseen_gene_ids.union(broken_gene_ids)
    inconsistent_gff = repaired_gff[repaired_gff['belong_gene'].isin(inconsistent_gene_id)]
    consistent_gff = repaired_gff[~repaired_gff['belong_gene'].isin(inconsistent_gene_id)]

    inconsistent_gff_path = os.path.join(saved_root, 'inconsistent_{}.gff3').format(postfix)
    consistent_gff_path = os.path.join(saved_root, 'consistent_{}.gff3').format(postfix)
    broken_block_gff_path = os.path.join(saved_root, 'broken_block_{}.gff3').format(postfix)
    unseen_block_gff_path = os.path.join(saved_root, 'unseen_block_{}.gff3').format(postfix)
    write_gff(inconsistent_gff,inconsistent_gff_path)
    write_gff(consistent_gff,consistent_gff_path)
    write_gff(broken_block_gff,broken_block_gff_path)
    write_gff(unseen_block_gff,unseen_block_gff_path)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="This program would use coding_seq and exon to " +
        "repair missing UTR, use coding_seq and UTR to repair exon\n" +
        ", and use exon to repair transcript, and use transcript to repair gene."
        + "Output data would be consistent data which are not changed.")
    parser.add_argument("-i","--input_path",help="Path of input GFF file",
                        required=True)
    parser.add_argument("-s","--saved_root",help="Root to save result",
                        required=True)
    parser.add_argument("-p", "--postfix", required=True)
    args = parser.parse_args()

    kwargs = vars(args)

    main(**kwargs)
