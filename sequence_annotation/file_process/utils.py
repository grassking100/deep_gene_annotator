import sys, os
import pandas as pd
from Bio import SeqIO
pd.set_option('mode.chained_assignment', 'raise')
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import CONSTANT_LIST, CONSTANT_DICT

CDS_TYPE = 'CDS'
UTR_TYPE = 'UTR'
GENE_TYPE = 'gene'
TRANSCRIPT_TYPE = 'transcript'
EXON_TYPE = 'exon'
INTRON_TYPE = 'intron'
INTERGENIC_REGION_TYPE = 'other'
EXON_SKIPPING = 'exon_skipping'
INTRON_RETENTION = 'intron_retention'
ALT_DONOR = 'alt_donor'
ALT_ACCEPTOR = 'alt_acceptor'
ALT_DONOR_SITE = 'alt_donor_site'
ALT_ACCEPTOR_SITE = 'alt_acceptor_site'
DONOR_ACCEPTOR_SITE = 'donor_acceptor_site'
SUBEXON_TYPES = CONSTANT_LIST([CDS_TYPE,UTR_TYPE])
ALT_STATUSES = CONSTANT_LIST([EXON_SKIPPING,INTRON_RETENTION,ALT_DONOR,ALT_ACCEPTOR,ALT_DONOR_SITE,ALT_ACCEPTOR_SITE,DONOR_ACCEPTOR_SITE])
BASIC_GFF_FEATURES = [GENE_TYPE,TRANSCRIPT_TYPE,EXON_TYPE,CDS_TYPE,UTR_TYPE]
BASIC_GENE_MAP = CONSTANT_DICT({GENE_TYPE: [EXON_TYPE, INTRON_TYPE],INTERGENIC_REGION_TYPE: [INTERGENIC_REGION_TYPE]})
BASIC_GENE_ANN_TYPES = CONSTANT_LIST([EXON_TYPE, INTRON_TYPE, INTERGENIC_REGION_TYPE])

BED_COLUMNS = CONSTANT_LIST([
    'chr', 'start', 'end', 'id', 'score', 'strand', 'thick_start', 'thick_end',
    'rgb', 'count', 'block_size', 'block_related_start'
])
GFF_COLUMNS = CONSTANT_LIST(['chr', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame','attribute'])

PLUS = 'plus'
MINUS = 'minus'

STRANDS = CONSTANT_LIST([PLUS, MINUS])

class InvalidStrandType(Exception):
    def __init__(self, strand_type=None):
        type_ = ""
        if strand_type is not None:
            type_ = ", " + str(strand_type) + ", "
        msg = "Strand type, {}, is not expected".format(type_)
        super().__init__(msg)


def get_gff_with_updated_attribute(gff):
    gff = gff.copy()
    columns = [c for c in gff.columns if c not in GFF_COLUMNS]
    attributes = []
    for column in columns:
        if column.lower() == 'id':
            key = 'ID'
        else:
            key = column.capitalize()
        values = gff[column]
        attribute = values.apply(lambda value: "{}={}".format(key, value))
        attributes += [attribute]
    if len(attributes) > 0:
        attribute = attributes[0]
        for attr in attributes[1:]:
            attribute = attribute + ";" + attr
        gff['attribute'] = attribute.replace(
            r"(;\w*=(None|\.))|(^\w*=(None|\.);)|(^\w*=(None|\.)$)",
            '',
            regex=True)
    return gff
        
def get_gff_with_intron(gff):
    gff = gff.copy()
    if 'parent' not in gff.columns:
        gff = get_gff_with_attribute(gff)
        #raise Exception("GFF file lacks 'parent' column")
    if (gff['feature']==INTRON_TYPE).any():
        gff = gff[gff['feature']!=INTRON_TYPE]
        #raise Exception("There already have intron in the GFF file")
    transcripts = gff[gff['feature']==TRANSCRIPT_TYPE]
    exons = gff[gff['feature']==EXON_TYPE]
    transcript_group = transcripts.groupby('id')
    exon_groups = exons.groupby('parent')
    introns = []
    for transcript_id, transcripts in transcript_group:
        intron_index = 1
        transcript = transcripts.iloc[0, :]
        intron_template = dict(transcript)
        intron_template['parent'] = transcript_id
        intron_template['feature'] = INTRON_TYPE
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
        #if update_attribute:
        introns = get_gff_with_updated_attribute(introns)
        gff = gff.append(introns,
                         ignore_index=True,
                         verify_integrity=True,
                         sort=True)
        gff = gff.reset_index(drop=True)
    return gff


def get_gff_with_intergenic_region(gff,region_table,chrom_source):#,update_attribute=True):
    transcripts = gff[gff['feature']==TRANSCRIPT_TYPE]
    intergenic_regions = []
    for chrom_id, group in region_table.groupby(chrom_source):
        for strand, length in zip(list(group['strand']),list(group['length'])):
            min_start = 1
            max_end = length + min_start - 1
            template = {
                'chr': chrom_id,
                'strand': strand,
                'feature': INTERGENIC_REGION_TYPE
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
    #if update_attribute:
    intergenic_regions = get_gff_with_updated_attribute(intergenic_regions)
    gff = gff.append(intergenic_regions,
                     ignore_index=True,
                     verify_integrity=True,
                     sort=True)
    gff = gff.drop_duplicates().reset_index(drop=True)
    return gff


def get_gff_with_belonging(gff,gene_types=None,transcript_types=None):
    gene_types = gene_types or [GENE_TYPE]
    transcript_types = transcript_types or [TRANSCRIPT_TYPE]
    if 'parent' not in gff.columns:
        raise Exception("GFF file lacks 'parent' column")
    gff = gff.copy()
    gff['belong_gene'] = None
    gff['belong_transcript'] = None
    gene_index = gff['feature'].isin(gene_types)
    gff.loc[gene_index, 'belong_gene'] = gff.loc[gene_index, 'id']
    transcript_index = gff['feature'].isin(transcript_types)
    gff.loc[transcript_index, 'belong_gene'] = gff.loc[transcript_index,'parent']
    gff.loc[transcript_index, 'belong_transcript'] = gff.loc[transcript_index,'id']
    transcript_convert_list = gff.loc[transcript_index,['id', 'parent']].to_dict('record')
    transcript_parent = {}
    for convert in transcript_convert_list:
        transcript_parent[convert['id']] = convert['parent']

    other_has_parent_index = (~gff['feature'].isin(gene_types + transcript_types)) & gff['parent'].notna()
    gff.loc[other_has_parent_index,'belong_transcript'] = gff.loc[other_has_parent_index, 'parent']
    belong_genes = [
        transcript_parent[item] for item in gff.loc[other_has_parent_index, 'belong_transcript']
    ]
    gff.loc[other_has_parent_index, 'belong_gene'] = belong_genes
    return gff


def gff_to_bed_command(gff_path, bed_path):
    src_root = os.path.dirname(__file__)
    to_bed_command = 'python3 {}/gff2bed.py -i {} -o {}'
    command = to_bed_command.format(src_root, gff_path, bed_path)
    os.system(command)


def validate_bed(bed,ignore_strand_check=False):
    if len(bed) > 0:
        if not ignore_strand_check:
            invalid_strands = set(bed['strand']) - set(['+', '-'])
            if len(invalid_strands) > 0:
                raise InvalidStrandType(invalid_strands)

        if ((bed['end'] - bed['start'] + 1) <= 0).any():
            raise Exception("Wrong transcript size")

        if 'start' in bed and (bed['start']<=0).any():
            raise Exception("Wrong transcript start")

        if 'end' in bed and (bed['end']<=0).any():
            raise Exception("Wrong transcript end")
            
        if 'thick_end' in bed.columns and 'thick_start' in bed.columns:
            if ((bed['thick_end'] - bed['thick_start'] + 1) < 0).any():
                raise Exception("Wrong coding size")

        if 'block_size' in bed.columns:
            block_size_list = list(bed['block_size'])
            for block_sizes in block_size_list:
                block_size_status = [
                    int(size) <= 0 for size in block_sizes.split(',')
                ]
                if any(block_size_status):
                    raise Exception("Wrong block size")

        if 'block_related_start' in bed.columns:
            block_related_start_list = list(bed['block_related_start'])
            for block_related_starts in block_related_start_list:
                site_status = [int(site) < 0 for site in block_related_starts.split(',')]
                if any(site_status):
                    raise Exception("Wrong block start size")


def validate_gff(gff,valid_features=None):
    if len(gff) > 0:
        invalid_strands = set(gff['strand']) - set(['+', '-'])
        if len(invalid_strands) > 0:
            raise InvalidStrandType(invalid_strands)
        if ((gff['end'] - gff['start'] + 1) <= 0).any():
            raise Exception("Wrong block size")
        if any(gff['start']<=0):
            raise Exception("Wrong block start")
        if any(gff['end']<=0):
            raise Exception("Wrong block end")
        if valid_features != False:
            invalid_features = set(gff['feature']) - set(valid_features)
            if len(invalid_features) > 0:
                raise Exception("Get invalid features {}".format(invalid_features))

def read_bed(path,ignore_strand_check=False):
    """
    Read bed data and convert from interbase coordinate system (ICS) to base coordinate system (BCS)
    For more information, please visit https://tidyomics.com/blog/2018/12/09/2018-12-09-the-devil-0-and-1-coordinate-system-in-genomics
    """
    bed = pd.read_csv(path, sep='\t', header=None, dtype={0: str,10:str,11:str})
    bed.columns = BED_COLUMNS[:len(bed.columns)]
    for name in ['start', 'end', 'thick_start', 'thick_end', 'count']:
        if name in bed.columns:
            bed[name] = bed[name]
            if name in ['start', 'thick_start']:
                bed[name] += 1
    validate_bed(bed,ignore_strand_check)
    if not ignore_strand_check:
        bed['five_end'] = bed['three_end'] = None
        plus_locs = bed['strand']=='+'
        minus_locs = bed['strand']=='-'
        bed.loc[plus_locs,'five_end'] = bed.loc[plus_locs,'start']
        bed.loc[plus_locs,'three_end'] = bed.loc[plus_locs,'end']
        bed.loc[minus_locs,'five_end'] = bed.loc[minus_locs,'end']
        bed.loc[minus_locs,'three_end'] = bed.loc[minus_locs,'start']
    return bed


def write_bed(bed, path,ignore_strand_check=False):
    """
    Convert bed data from base coordinate system(BCS) to interbase coordinate system（ICS) and write to file
    For more information, please visit https://tidyomics.com/blog/2018/12/09/2018-12-09-the-devil-0-and-1-coordinate-system-in-genomics
    """
    validate_bed(bed,ignore_strand_check)
    columns = []
    for name in BED_COLUMNS:
        if name in bed.columns:
            columns.append(name)
    bed = bed[columns].astype(str)
    for name in ['start', 'end', 'thick_start', 'thick_end', 'count']:
        if name in bed.columns:
            bed[name] = bed[name].astype(int)
            if name in ['start', 'thick_start']:
                bed[name] -= 1
    bed.to_csv(path, sep='\t', index=None, header=None)


def read_gff(path,with_attr=False,valid_features=None):
    if valid_features is None:
        valid_features = BASIC_GFF_FEATURES
    gff = pd.read_csv(path,sep='\t',header=None,dtype=str,
                      names=list(range(len(GFF_COLUMNS))))
    gff.columns = GFF_COLUMNS
    gff = gff[~gff['chr'].str.startswith('#')]
    int_columns = ['start', 'end']
    gff.loc[:, int_columns] = gff[int_columns].astype(float).astype(int)
    validate_gff(gff,valid_features=valid_features)
    if with_attr:
        gff = get_gff_with_attribute(gff)
    return gff


def write_gff(gff, path,valid_features=None,update_attr=False):
    if valid_features is None:
        valid_features = BASIC_GFF_FEATURES
    validate_gff(gff,valid_features=valid_features)
    if update_attr:
        gff = get_gff_with_updated_attribute(gff)
    with open(path, 'w') as fp:
        fp.write("##gff-version 3\n")
        gff[GFF_COLUMNS].to_csv(fp, header=None, sep='\t', index=None)


def get_gff_item_with_attribute(item, split_attr=None):
    split_attr = split_attr or []
    attributes = item['attribute'].split(';')
    attribute_dict = {}
    for attribute in attributes:
        if len(attribute)>1:
            lhs, rhs = attribute.split('=')
            lhs = lhs.lower().replace('-', '_')
            if lhs in split_attr:
                rhs = rhs.split(',')
            attribute_dict[lhs] = rhs
    copied = dict(item)
    copied.update(attribute_dict)
    return copied


def get_gff_with_attribute(gff, split_attr=None):
    df_dict = gff.to_dict('record')
    data = []
    for item in df_dict:
        data += [get_gff_item_with_attribute(item, split_attr)]
    gff = pd.DataFrame.from_dict(data)
    gff = gff.where(pd.notnull(gff), None)
    return gff


def get_gff_with_updated_attribute(gff):
    gff = gff.copy()
    columns = [c for c in gff.columns if c not in GFF_COLUMNS]
    attributes = []
    for column in columns:
        if column.lower() == 'id':
            key = 'ID'
        else:
            key = column.capitalize()
        values = gff[column]
        attribute = values.apply(lambda value: "{}={}".format(key, value))
        attributes += [attribute]
    if len(attributes) > 0:
        attribute = attributes[0]
        for attr in attributes[1:]:
            attribute = attribute + ";" + attr
        gff['attribute'] = attribute.replace(
            r"(;\w*=(None|\.))|(^\w*=(None|\.);)|(^\w*=(None|\.)$)",
            '',
            regex=True)
    return gff


def dupliacte_gff_by_parent(gff):
    if 'parent' not in gff.columns:
        raise Exception("GFF file lacks 'parent' column")

    valid_parents = [p for p in gff['parent'] if p is not None]
    if len(valid_parents) > 0:
        if not isinstance(valid_parents[0], list):
            raise Exception("GFF's 'parent' data type should be list")

    preprocessed = []
    for item in gff.to_dict('record'):
        parents = item['parent']
        if parents is not None:
            for parent in parents:
                item_ = dict(item)
                item_['parent'] = str(parent)
                preprocessed.append(item_)
        else:
            preprocessed.append(item)
    gff = pd.DataFrame.from_dict(preprocessed)
    return gff


def read_fai(path):
    chrom_info = pd.read_csv(path, header=None, sep='\t')
    chrom_id, chrom_length = chrom_info[0], chrom_info[1]
    chrom_info = {}
    for id_, length in zip(chrom_id, chrom_length):
        chrom_info[str(id_)] = length
    return chrom_info


def read_fasta(path, check_unique_id=True):
    """Read fasta file and return dictionary of sequneces"""
    data = {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as file:
        fasta_sequences = SeqIO.parse(file, 'fasta')
        for fasta in fasta_sequences:
            seq = str(fasta.seq)
            name = fasta.description
            if check_unique_id and name in data:
                raise Exception("Duplicate id {}".format(name))
            data[name] = seq
    return data


def write_fasta(seqs,path):
    """Read dictionary of sequneces into fasta file"""
    with open(path, "w") as file:
        for id_, seq in seqs.items():
            file.write(">" + id_ + "\n")
            file.write(seq + "\n")

            
def get_gff_with_feature_coord(gff):
    gff = gff.copy()
    part_gff = gff[['feature', 'chr', 'strand', 'start', 'end']]
    feature_coord = part_gff.apply(
        lambda x: '_'.join([str(item) for item in x]), axis=1)
    gff = gff.assign(feature_coord=feature_coord)
    return gff


def rename_fasta(input_path,output_path):
    command = "sed  '/>/ s/^>Chr/>/g' {}  > {}".format(input_path,output_path)
    os.system(command)


def create_fai(path):
    os.system("rm {}.fai".format(path))
    os.system("samtools faidx {}".format(path))

