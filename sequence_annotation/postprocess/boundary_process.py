import pandas as pd
from ..utils.exception import InvalidStrandType
from ..utils.utils import find_substr
from ..genome_handler.sequence import AnnSequence, PLUS
from ..genome_handler.region_extractor import RegionExtractor


def get_motifs(path, first_n=None):
    signal = pd.read_csv(path, sep='\t')
    signal = signal.sort_values('count', ascending=False)
    motifs = list(signal['motif'])
    if first_n is not None:
        motifs = motifs[:first_n]
    return motifs


def get_splicing_regex(path, first_n=None):
    first_n = first_n or 1
    motifs = get_motifs(path, first_n)
    motif_regex = '|'.join(motifs)
    return motif_regex


def find_closest_site(pred_site, ann_sites, dist):
    """Find pred_site's most closest annotated sites in specific distance,
    if it not found then it will return pred_site itself

    Parameters:
    ----------
    pred_site : int
        Location of predicted site
    ann_sites : iterable
        Locations of annotated sites
    dist : numeric
        Valid distance from pred_site to ann_sites

    Returns:
    ----------
    tuple (numeric, numeric, bool)
          1. Clostest site
          2. Distance between pred_site and closest site
          3. Is annotated site found in specific distance or not
    """
    diff = [abs(ann_site - pred_site) for ann_site in ann_sites]
    fixed_pred_site = None
    fixed_diff = None
    for diff_, ann_site in zip(diff, ann_sites):
        if diff_ <= dist:
            if fixed_pred_site is None or diff_ < fixed_diff:
                fixed_pred_site = ann_site
                fixed_diff = diff_
    in_dist = fixed_pred_site is not None
    if fixed_pred_site is None:
        fixed_pred_site = pred_site
        fixed_diff = 0
    return fixed_pred_site, fixed_diff, in_dist


def get_splice_pairs(gff):
    """Get splice pairs in order of donor site and acceptor site

    Parameters:
    ----------
    gff : pd.DataFrame
        GFF data in pd.DataFrame format

    Returns:
    ----------
    list (tuple)
        List of paired sites of donor site and acceptor site in one based
    """
    strands = set(gff['strand'])
    if strands != set(['+']):
        raise InvalidStrandType(strands)

    introns = gff[gff['feature'] == 'intron']
    pairs = list(zip(list(introns['start']), list(introns['end'])))
    return pairs


def get_splice_pair_statuses(splice_pairs, ann_donor_sites, ann_acceptor_sites,
                             donor_distance, acceptor_distance):
    """Determite splice pair status by its distance between answer
    Parameters:
    ----------
    splice_pairs : list (tuple)
        List of paired sites of donor site and acceptor site in one based
    ann_donor_sites : list (int)
        Location of donor sites in one based
    ann_acceptor_sites : list (int)
        Location of acccept sites in one based
    donor_distance : int
        Valid donor site distance
    acceptor_distance : int
        Valid acceptor site distance
    Returns:
    ----------
    dict
    """
    splice_pair_statuses = []
    for splice_pair in splice_pairs:
        donor_site, acceptor_site = splice_pair
        fixed_donor_site, _, donor_in_dist = find_closest_site(
            donor_site, ann_donor_sites, donor_distance)
        fixed_acceptor_site, _, acceptor_in_dist = find_closest_site(
            acceptor_site, ann_acceptor_sites, acceptor_distance)
        status = {'donor': {'location': donor_site, 'valid': False},
                  'acceptor': {'location': acceptor_site, 'valid': False}
                  }
        if donor_in_dist:
            status['donor']['location'] = fixed_donor_site
            status['donor']['valid'] = True
        if acceptor_in_dist:
            status['acceptor']['location'] = fixed_acceptor_site
            status['acceptor']['valid'] = True
        if status['donor']['location'] < status['acceptor']['location']:
            splice_pair_statuses.append(status)
    return splice_pair_statuses


def get_valid_intron_boundary(rna_boundary, splice_pair_statuses):
    """Get list of intron boundarys based on gene boundary and intron boundarys
    Parameters:
    ----------
    rna_boundary : tuple
        Paired sites of RNA's start site and end site in one based
    splice_pair_statuses : dcit
        The get_splice_pair_statuses is generated by get_splice_pair_statuses

    Returns:
    ----------
    list (tuple)
        List of paired sites of intron's start site and end site in one based
    """
    not_terminal_statuses = []
    rna_start, rna_end = rna_boundary
    rna_length = rna_end - rna_start + 1
    for status in splice_pair_statuses:
        start = status['donor']['location']
        end = status['acceptor']['location']
        if rna_start < start and end < rna_end:
            not_terminal_statuses.append(status)

    valid_introns = set()
    valid_donor_introns = set()
    valid_acceptor_introns = set()
    for status in not_terminal_statuses:
        is_donor_valid = status['donor']['valid']
        is_acceptor_valid = status['acceptor']['valid']
        donor = status['donor']['location']
        acceptor = status['acceptor']['location']
        id_ = "{}_{}".format(donor, acceptor)
        if is_donor_valid and is_acceptor_valid:
            valid_introns.add(id_)
        if is_donor_valid:
            valid_donor_introns.add(id_)
        if is_acceptor_valid:
            valid_acceptor_introns.add(id_)

    for valid_donor_intron in valid_donor_introns:
        valid_donor, valid_partner_acceptor = valid_donor_intron.split('_')
        valid_donor = int(valid_donor)
        valid_partner_acceptor = int(valid_partner_acceptor)
        for valid_acceptor_intron in valid_acceptor_introns:
            valid_partner_donor, valid_acceptor = valid_acceptor_intron.split(
                '_')
            valid_partner_donor = int(valid_partner_donor)
            valid_acceptor = int(valid_acceptor)
            # Merge intron
            if (valid_acceptor - valid_donor + 1 >
                    0) and ((valid_partner_donor - valid_partner_acceptor - 1) <= 0):
                id_ = "{}_{}".format(valid_donor, valid_acceptor)
                #print("Create merge inton {}".format(id_))
                valid_introns.add(id_)

    ann_seq = AnnSequence(['intron'], rna_length)
    ann_seq.strand = PLUS
    for valid_intron in valid_introns:
        start, end = valid_intron.split('_')
        start = int(start) - rna_start
        end = int(end) - rna_start
        ann_seq.set_ann("intron", 1, start, end)
    blocks = RegionExtractor().extract(ann_seq)
    valid_intron_boundarys_ = []
    for block in blocks:
        valid_intron_boundarys_.append(
            (block.start + rna_start, block.end + rna_start))
    return valid_intron_boundarys_


def get_exon_boundary(rna_boundary, intron_boundarys):
    """Get list of exon boundarys based on gene boundary and intron boundarys

    Parameters:
    ----------
    rna_boundary : list (tuple)
        List of paired sites of RNA's start site and end site in one based
    intron_boundarys : list (tuple)
        The intron_boundarys is generated by get_valid_intron_boundary
    Returns:
    ----------
    list (tuple)
        List of paired sites of exon's start site and end site in one based
    """
    intron_boundarys = sorted(intron_boundarys, key=lambda x: x[0])
    exon_start = rna_boundary[0]
    exon_boundarys = []
    for intron_boundary in intron_boundarys:
        intron_start, intron_end = intron_boundary
        exon_end = intron_start - 1
        if exon_start <= exon_end:
            exon_boundarys.append((exon_start, exon_end))
        else:
            raise Exception("Exon start is larger than exon end")
        exon_start = intron_end + 1

    exon_end = rna_boundary[1]
    if exon_start <= exon_end:
        exon_boundarys.append((exon_start, exon_end))
    return exon_boundarys


def guess_boundarys(seq, donor_site_pattern=None, acceptor_site_pattern=None,
                    donor_site_index_shift=None, acceptor_site_index_shift=None):
    """Get list of guessed intron boundarys based on donor site pattern and acceptor site pattern

    Parameters:
    ----------
    seq : str
        DNA sequence which its direction is 5' to 3'
    donor_site_pattern : str (default : GT)
        Regular expression of donor site
    acceptor_site_pattern : str (default : AG)
        Regular expression of acceptor site
    donor_site_index_shift : int (default : 0)
        Shift value of donor site index
    acceptor_site_index_shift : int (default : 1)
        Shift value of acceptor site index

    Returns:
    ----------
    list (tuple)
        List of guessed intron boundarys
    """
    donor_site_index_shift = donor_site_index_shift or 0
    acceptor_site_index_shift = acceptor_site_index_shift or 1
    donor_site_pattern = donor_site_pattern or 'GT'
    acceptor_site_pattern = acceptor_site_pattern or 'AG'
    ann_donor_sites = find_substr(
        donor_site_pattern, seq, donor_site_index_shift + 1)
    ann_acceptor_sites = find_substr(
        acceptor_site_pattern,
        seq,
        acceptor_site_index_shift + 1)
    type_ = {}
    for site in ann_donor_sites:
        type_[site] = 'D'
    for site in ann_acceptor_sites:
        type_[site] = 'A'
    sites = sorted(ann_donor_sites + ann_acceptor_sites)
    boundarys = []
    previous_site = None
    for site in sites:
        if previous_site is None:
            if type_[site] == 'D':
                previous_site = site
        else:
            if type_[site] == 'A':
                if previous_site < site:
                    boundarys.append((previous_site, site))
                    previous_site = None
    return boundarys
