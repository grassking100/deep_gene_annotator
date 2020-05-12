import sys
import os
import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser
from sequence_annotation.utils.utils import write_gff, read_bed, get_gff_with_updated_attribute
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict

strand_convert = {'plus': '+', 'minus': '-', '+': '+', '-': '-'}


def bed_item2gff_item(item, id_convert_dict):
    # input zero based data, return one based data
    gff_items = []
    thick_start = item['thick_start'] + 1
    thick_end = item['thick_end'] + 1
    mRNA_id = item['id']
    gene_id = id_convert_dict[mRNA_id]
    strand = strand_convert[item['strand']]
    basic_block = {'chr': item['chr'], 'strand': strand, 'source': '.',
                   'score': item['score'], 'frame': '.'}
    gene = dict(basic_block)
    gene['start'], gene['end'] = item['start'] + 1, item['end'] + 1
    gene['feature'] = 'gene'
    gene['id'] = gene_id
    mRNA = dict(gene)
    mRNA['feature'] = 'mRNA'
    mRNA['id'] = mRNA_id
    mRNA['parent'] = gene_id
    gff_items.append(gene)
    gff_items.append(mRNA)
    exons = []
    for index in range(item['count']):
        abs_start = item['block_related_start'][index] + gene['start']
        exon = dict(basic_block)
        exon['start'] = abs_start
        exon['end'] = abs_start + item['block_size'][index] - 1
        exon['feature'] = 'exon'
        exon['parent'] = mRNA_id
        exons.append(exon)

    for index, exon in enumerate(exons):
        gff_items.append(exon)
        # Create intron info
        if index < len(exons) - 1:
            next_exon = exons[index + 1]
            intron = dict(exon)
            intron['start'] = exon['end'] + 1
            intron['end'] = next_exon['start'] - 1
            intron['feature'] = 'intron'
            gff_items.append(intron)

        # If Coding region is exist, create UTR or CDS info
        if thick_start <= thick_end:
            cds_site = []
            if exon['start'] <= thick_start <= exon['end']:
                cds_site.append(thick_start)
            if exon['start'] <= thick_end <= exon['end']:
                cds_site.append(thick_end)
            if len(cds_site) == 0:
                whole_block = dict(exon)
                if thick_start <= exon['start'] and exon['end'] <= thick_end:
                    whole_block['feature'] = 'CDS'
                else:
                    whole_block['feature'] = 'UTR'
                gff_items.append(whole_block)
            elif len(cds_site) == 2:
                utr = dict(exon)
                utr['feature'] = 'UTR'
                utr_1 = dict(utr)
                utr_2 = dict(utr)
                utr_1['end'], utr_2['start'] = thick_start - 1, thick_end + 1
                cds = dict(exon)
                cds['feature'] = 'CDS'
                cds['start'], cds['end'] = thick_start, thick_end
                gff_items.append(cds)
                gff_items.append(utr_1)
                gff_items.append(utr_2)
            else:
                if exon['start'] <= thick_start <= exon['end']:
                    utr = dict(exon)
                    utr['feature'] = 'UTR'
                    utr['end'] = thick_start - 1
                    cds = dict(exon)
                    cds['feature'] = 'CDS'
                    cds['start'], cds['end'] = thick_start, exon['end']
                    gff_items.append(cds)
                    gff_items.append(utr)
                else:
                    utr = dict(exon)
                    utr['feature'] = 'UTR'
                    utr['start'] = thick_end + 1
                    cds = dict(exon)
                    cds['feature'] = 'CDS'
                    cds['start'], cds['end'] = exon['start'], thick_end
                    gff_items.append(cds)
                    gff_items.append(utr)
        # Create UTR info
        else:
            whole_block = dict(exon)
            whole_block['feature'] = 'UTR'
            gff_items.append(whole_block)

    return gff_items


def simple_bed2gff(bed, feature=None):
    feature = feature or '.'
    gff_items = []
    for item in bed.to_dict('record'):
        strand = strand_convert[item['strand']]
        basic_block = dict(item)
        basic_block.update({'strand': strand, 'frame': '.',
                            'feature': feature, 'source': '.'})
        gff_items.append(basic_block)
    gff = pd.DataFrame.from_dict(gff_items)
    gff = gff[(gff['end'] - gff['start'] + 1) > 0]
    del gff['five_end']
    del gff['three_end']
    gff = get_gff_with_updated_attribute(gff)
    return gff


def bed2gff(bed, id_convert_dict):
    parser = BedInfoParser()
    bed = parser.parse(bed)
    gff_items = {}
    basic_items = []
    for item in bed:
        basic_items.append(bed_item2gff_item(item, id_convert_dict))
    for basic_item in basic_items:
        gene_id = basic_item[0]['id']
        if gene_id not in gff_items.keys():
            gff_items[gene_id] = [basic_item[0]]
        gff_items[gene_id] += basic_item[1:]
    gff_list = []
    for list_ in gff_items.values():
        gff_list += list_
    gff = pd.DataFrame.from_dict(gff_list)
    gff = gff[(gff['end'] - gff['start'] + 1) > 0]
    gff = get_gff_with_updated_attribute(gff)
    return gff


def main(bed_path, gff_output, feature=None,
         id_table_path=None, simple_mode=False):
    bed = read_bed(bed_path)
    if simple_mode:
        gff = simple_bed2gff(bed, feature)
    else:
        if id_table_path is None:
            raise Exception("Please provide id_table_path")
        id_convert_dict = get_id_convert_dict(id_table_path)
        gff = bed2gff(bed, id_convert_dict)
    write_gff(gff, gff_output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--bed_path", required=True)
    parser.add_argument("-o", "--gff_output", required=True)
    parser.add_argument(
        "-t",
        "--id_table_path",
        help="Id table about transcript and gene conversion")
    parser.add_argument("--simple_mode", action='store_true')
    parser.add_argument("--feature", help='Set feature if it is simple mode')
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
