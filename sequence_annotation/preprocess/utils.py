import sys, os
import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import BED_COLUMNS, GFF_COLUMNS, CONSTANT_LIST
from sequence_annotation.utils.utils import get_gff_with_updated_attribute, get_file_name

preprocess_src_root = os.path.dirname(__file__)

GENE_TYPES = CONSTANT_LIST([
    'gene', 'transposable_element', 'transposable_element_gene', 'pseudogene'
])
RNA_TYPES = CONSTANT_LIST([
    'mRNA', 'pseudogenic_tRNA', 'pseudogenic_transcript', 'antisense_lncRNA',
    'lnc_RNA', 'antisense_RNA', 'transcript_region', 'transposon_fragment',
    'miRNA_primary_transcript', 'tRNA', 'snRNA', 'ncRNA', 'snoRNA', 'rRNA',
    'transcript','pre_miRNA'
])
EXON_TYPES = CONSTANT_LIST(['exon', 'pseudogenic_exon'])
INTRON_TYPES = CONSTANT_LIST(['intron'])
SUBEXON_TYPES = CONSTANT_LIST(
    ['five_prime_UTR', 'three_prime_UTR', 'CDS', 'UTR'])
PROTEIN_TYPES = CONSTANT_LIST(['protein'])
UORF_TYPES = CONSTANT_LIST(['uORF'])
MIRNA_TPYES = CONSTANT_LIST(['miRNA'])

EXON_SKIPPING = 'exon_skipping'
INTRON_RETENTION = 'intron_retention'
ALT_DONOR = 'alt_donor'
ALT_ACCEPTOR = 'alt_acceptor'
ALT_DONOR_SITE = 'alt_donor_site'
ALT_ACCEPTOR_SITE = 'alt_acceptor_site'
ALT_REGIONS = CONSTANT_LIST([ALT_DONOR, ALT_ACCEPTOR])


def classify_data_by_id(bed, selected_ids, id_convert=None):
    all_ids = list(set(bed['id']))
    selected_ids = set(selected_ids)
    if id_convert is not None:
        selected_gene_ids = set(id_convert[id_] for id_ in selected_ids)
        id_info = pd.DataFrame(all_ids)
        id_info.columns = ['ref_id']
        gene_id = []
        for id_ in id_info['ref_id']:
            if id_ in id_convert.keys():
                gene_id.append(id_convert[id_])
            else:
                gene_id.append(id_)
        id_info = id_info.assign(gene_id=pd.Series(gene_id).values)
        match_status = id_info['gene_id'].isin(selected_gene_ids)
        want_status = id_info['ref_id'].isin(selected_ids)
        id_info['status'] = 'unwant'
        id_info.loc[match_status & want_status, 'status'] = 'want'
        id_info.loc[match_status & ~want_status, 'status'] = 'discard'
        want_id = id_info[id_info['status'] == 'want']['ref_id']
        unwant_id = id_info[id_info['status'] == 'unwant']['ref_id']
    else:
        id_info = pd.DataFrame(all_ids, columns=['id'])
        want_status = id_info['id'].isin(selected_ids)
        want_id = id_info[want_status]['id']
        unwant_id = id_info[~want_status]['id']
    want_bed = bed[bed['id'].isin(want_id)].drop_duplicates()
    unwant_bed = bed[bed['id'].isin(unwant_id)].drop_duplicates()
    return want_bed, unwant_bed


def simply_coord(bed):
    bed = bed[BED_COLUMNS[:6]]
    bed = bed.assign(id=pd.Series('.', index=bed.index))
    bed = bed.assign(score=pd.Series('.', index=bed.index))
    bed = bed.drop_duplicates()
    return bed


def simply_coord_with_gene_id(bed, id_convert=None):
    bed = bed[BED_COLUMNS[:6]]
    if id_convert is not None:
        gene_ids = [id_convert[id_] for id_ in bed['id']]
        bed = bed.assign(id=pd.Series(gene_ids).values)
    bed = bed.assign(score=pd.Series('.', index=bed.index))
    bed = bed.drop_duplicates()
    return bed


def _get_feature_coord(gff_item):
    part = [
        str(gff_item[type_])
        for type_ in ['feature', 'chr', 'strand', 'start', 'end']
    ]
    feature_coord = '_'.join(part)
    return feature_coord


def get_gff_with_intron(gff, update_attribute=True):
    if 'parent' not in gff.columns:
        raise Exception("GFF file lacks 'parent' column")
    if gff['feature'].isin(INTRON_TYPES).any():
        raise Exception("There already have intron in the GFF file")
    gff = gff.copy()
    transcripts = gff[gff['feature'].isin(RNA_TYPES)]
    exons = gff[gff['feature'].isin(EXON_TYPES)]
    transcript_group = transcripts.groupby('id')
    exon_groups = exons.groupby('parent')
    introns = []
    for transcript_id, transcripts in transcript_group:
        intron_index = 1
        transcript = transcripts.iloc[0, :]
        intron_template = dict(transcript)
        intron_template['parent'] = transcript_id
        intron_template['feature'] = 'intron'
        #There is at least one exon in the transcript
        if transcript_id in exon_groups.groups.keys():
            exon_group = exon_groups.get_group(transcript_id)
            #There is intron at start
            if transcript['start'] < exon_group['start'].min():
                intron = dict(intron_template)
                intron['end'] = exon_group['start'].min() - 1
                intron['id'] = '{}_intron_{}'.format(transcript_id,
                                                     intron_index)
                introns.append(intron)
                intron_index += 1
            #There is intron at end
            if transcript['end'] > exon_group['end'].max():
                intron = dict(intron_template)
                intron['start'] = exon_group['end'].max() + 1
                intron['id'] = '{}_intron_{}'.format(transcript_id,
                                                     intron_index)
                introns.append(intron)
                intron_index += 1
            exon_sites = sorted(exon_group['start'].tolist() +
                                exon_group['end'].tolist())
            for index in range(1, len(exon_sites) - 1, 2):
                intron = dict(intron_template)
                intron['id'] = '{}_intron_{}'.format(transcript_id,
                                                     index + intron_index - 1)
                intron['start'] = exon_sites[index] + 1
                intron['end'] = exon_sites[index + 1] - 1
                introns.append(intron)
        #There is not exon in the transcript
        else:
            intron = dict(intron_template)
            intron['id'] = '{}_intron_{}'.format(transcript_id, intron_index)
            introns.append(intron)

    if len(introns) > 0:
        introns = pd.DataFrame.from_dict(introns)
        if update_attribute:
            introns = get_gff_with_updated_attribute(introns)
        gff = gff.append(introns,
                         ignore_index=True,
                         verify_integrity=True,
                         sort=True)
        gff = gff.reset_index(drop=True)
    return gff


def get_gff_with_intergenic_region(gff,
                                   region_table,
                                   chrom_source,
                                   update_attribute=True):
    transcripts = gff[gff['feature'].isin(RNA_TYPES)]
    intergenic_regions = []
    for chrom_id, group in region_table.groupby(chrom_source):
        for strand, length in zip(list(group['strand']),
                                  list(group['length'])):
            #print(chrom_id,strand)
            min_start = 1
            max_end = length + min_start - 1
            template = {
                'chr': chrom_id,
                'strand': strand,
                'feature': 'intergenic region'
            }
            items = transcripts[(transcripts['chr'] == chrom_id)
                                & (transcripts['strand'] == strand)]
            region_sites = (items['start'] - 1).tolist() + (items['end'] +
                                                            1).tolist()
            region_sites = sorted(list(set(region_sites)))
            if len(region_sites) > 0:
                if min(region_sites) != min_start:
                    region_sites = [min_start] + region_sites
                if max(region_sites) != max_end:
                    region_sites = region_sites + [max_end]
            else:
                region_sites = [min_start, max_end]
            for index in range(0, len(region_sites) - 1, 2):
                intergenic_region = dict(template)
                lhs_site = region_sites[index]
                rhs_site = region_sites[index + 1]
                intergenic_region['start'] = min(lhs_site, rhs_site)
                intergenic_region['end'] = max(lhs_site, rhs_site)
                for column in GFF_COLUMNS:
                    if column not in intergenic_region:
                        intergenic_region[column] = '.'
                intergenic_regions.append(intergenic_region)
    intergenic_regions = pd.DataFrame.from_dict(intergenic_regions)
    if update_attribute:
        intergenic_regions = get_gff_with_updated_attribute(intergenic_regions)
    gff = gff.append(intergenic_regions,
                     ignore_index=True,
                     verify_integrity=True,
                     sort=True)
    gff = gff.drop_duplicates().reset_index(drop=True)
    return gff


def get_gff_with_belonging(gff):
    if 'parent' not in gff.columns:
        raise Exception("GFF file lacks 'parent' column")
    gff = gff.copy()
    gff['belong_gene'] = None
    gff['belong_transcript'] = None
    gene_index = gff['feature'].isin(GENE_TYPES)
    gff.loc[gene_index, 'belong_gene'] = gff.loc[gene_index, 'id']
    transcript_index = gff['feature'].isin(RNA_TYPES)
    gff.loc[transcript_index, 'belong_gene'] = gff.loc[transcript_index,
                                                       'parent']
    gff.loc[transcript_index, 'belong_transcript'] = gff.loc[transcript_index,
                                                             'id']
    transcript_convert_list = gff.loc[transcript_index,
                                      ['id', 'parent']].to_dict('record')
    transcript_parent = {}
    for convert in transcript_convert_list:
        transcript_parent[convert['id']] = convert['parent']

    other_has_parent_index = (
        ~gff['feature'].isin(GENE_TYPES + RNA_TYPES)) & gff['parent'].notna()
    gff.loc[other_has_parent_index,
            'belong_transcript'] = gff.loc[other_has_parent_index, 'parent']
    belong_genes = [
        transcript_parent[item]
        for item in gff.loc[other_has_parent_index, 'belong_transcript']
    ]
    gff.loc[other_has_parent_index, 'belong_gene'] = belong_genes
    return gff


def gff_to_bed_command(gff_path, bed_path):
    to_bed_command = 'python3 {}/gff2bed.py -i {} -o {}'
    command = to_bed_command.format(preprocess_src_root, gff_path, bed_path)
    os.system(command)


def get_data_names(split_root):
    split_table = pd.read_csv(os.path.join(
        split_root, 'split_table.csv')).to_dict('record')
    paths = {}
    for item in split_table:
        training_name = get_file_name(item['training_path'])
        val_name = get_file_name(item['validation_path'])
        testing_name = get_file_name(item['testing_path'])
        name = "{}_{}".format(training_name, val_name)
        paths[name] = {
            'validation': val_name,
            'testing': testing_name,
            'training': training_name
        }
    return paths


def read_region_table(path, calculate_length=True):
    """Get region table about regions"""
    df = pd.read_csv(path,
                     sep='\t',
                     dtype={
                         'chr': str,
                         'start': int,
                         'end': int
                     })
    if calculate_length:
        df['length'] = df['end'] - df['start'] + 1
    return df
