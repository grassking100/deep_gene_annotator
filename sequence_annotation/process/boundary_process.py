import re
from ..preprocess.utils import RNA_TYPES
from ..genome_handler.sequence import AnnSequence
from ..genome_handler.ann_seq_processor import get_background

def find_substr(regex,string,show_start_index=True):
    """Find indice of matched text in string
    
    Parameters:
    ----------
    regex : str
        Regular expression
    string : str
        String to be searched
    show_start_index : bool (default: True)
        Return start index of matched text in string, of
        return end index of matched text in string
        
    Returns:
    ----------
    list (int)
        List of indice
    """
    iter_ = re.finditer(regex, string)
    if show_start_index:
        indice = [m.start() for m in iter_]
    else:    
        indice = [m.end()-1 for m in iter_]
    return indice
    
def find_clocest_site(pred_site,ann_sites,dist):
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
    diff = [abs(ann_site-pred_site) for ann_site in ann_sites]
    fixed_pred_site = None
    fixed_diff = None
    for diff_, ann_site in zip(diff,ann_sites):
        if diff_ <= dist:
            if fixed_pred_site is None or diff_ < fixed_diff:
                fixed_pred_site = ann_site
                fixed_diff = diff_
    in_dist = fixed_pred_site is not None
    if fixed_pred_site is None:
        fixed_pred_site = pred_site
        fixed_diff = 0
    return fixed_pred_site,fixed_diff,in_dist
    
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
    if (gff['strand'] != '+').any():
        raise Exception("Invalid strand")

    introns = gff[gff['feature'] == 'intron']
    pairs = list(zip(list(introns['start']),list(introns['end'])))
    return pairs
    
def fix_splice_pairs(splice_pairs,ann_donor_sites,ann_acceptor_sites,dist):
    """Fix splice pairs by ann_donor_sites and ann_acceptor_sites
        
    Parameters:
    ----------
    splice_pairs : list (tuple)
        List of paired sites of donor site and acceptor site in one based
    ann_donor_sites : list (int)
        Location of donor sites in one based
    ann_acceptor_sites : list (int)
        Location of acccept sites in one based
    dist : int
        Valid distance

    Returns:
    ----------
    list (tuple)
        List of fixed paired sites of donor site and acceptor site in one based
    """
    fix_splice_pairs_ = []
    for splice_pair in splice_pairs:
        donor_site,acceptor_site = splice_pair
        fixed_donor_site,donor_diff,donor_in_dist = find_clocest_site(donor_site,ann_donor_sites,dist)
        if donor_in_dist:
            fixed_acceptor_site,_,acceptor_in_dist = find_clocest_site(acceptor_site,ann_acceptor_sites,dist)
            if acceptor_in_dist and fixed_donor_site < fixed_acceptor_site:
                fix_splice_pairs_.append((fixed_donor_site,fixed_acceptor_site))
    return fix_splice_pairs_

def get_exon_boundary(rna_boundary,intron_boundarys):
    """Get list of exon boundarys based on gene boundary and intron boundarys
        
    Parameters:
    ----------
    rna_boundary : list (tuple)
        List of paired sites of RNA's start site and end site in one based
    intron_boundarys : list (tuple)
        List of paired sites of intron's start site and end site in one based

    Returns:
    ----------
    list (tuple)
        List of paired sites of exon's start site and end site in one based
    """
    intron_boundarys = sorted(intron_boundarys,key=lambda x: x[0])    
                        
    exon_start = rna_boundary[0]
    exon_boundarys = []
    for intron_boundary in intron_boundarys:
        intron_start, intron_end = intron_boundary
        exon_end = intron_start - 1
        if exon_start < exon_end:
            exon_boundarys.append((exon_start,exon_end))
        exon_start = intron_end + 1
    
    exon_end = rna_boundary[1]
    if exon_start < exon_end:
        exon_boundarys.append((exon_start,exon_end))
    return exon_boundarys

def get_fixed_intron_boundary(rna_boundary,intron_boundarys):
    """Get list of intron boundarys based on gene boundary and intron boundarys,
    introns will be discarded if it will cause wrong annotation
        
    Parameters:
    ----------
    rna_boundary : list (tuple)
        List of paired sites of RNA's start site and end site in one based
    intron_boundarys : list (tuple)
        List of paired sites of intron's start site and end site in one based

    Returns:
    ----------
    list (tuple)
        List of paired sites of intron's start site and end site in one based
    """
    intron_boundarys = sorted(intron_boundarys,key=lambda x: x[0])    
    is_intron_valid = {}
    for index, intron_boundary in enumerate(intron_boundarys):
        id_ = '{}_{}'.format(*intron_boundary)
        if index == 0:
            is_intron_valid[id_] =  rna_boundary[0] < intron_boundary[0]
        else:
            previous_intron_boundarys = intron_boundarys[index-1]
            is_intron_valid[id_] = True
            if (intron_boundary[0] - previous_intron_boundarys[1] - 1) <= 0:
                previous_id = '{}_{}'.format(*previous_intron_boundarys)
                
                is_intron_valid[previous_id] = False
                is_intron_valid[id_] = False
                
            if index == len(intron_boundarys)-1 and intron_boundary[1] >= rna_boundary[1]:
                is_intron_valid[id_] =  False
                
    valid_intron_boundarys = []
    for intron_boundary in intron_boundarys:
        id_ = '{}_{}'.format(*intron_boundary)
        if is_intron_valid[id_]:
            valid_intron_boundarys.append(intron_boundary)
    
    return valid_intron_boundarys

def guess_boundarys(seq,donor_site_pattern=None,acceptor_site_pattern=None):
    """Get list of guessed intron boundarys based on donor site pattern and acceptor site pattern
        
    Parameters:
    ----------
    seq : str
        DNA sequence which its direction is 5' to 3'
    donor_site_pattern : str (default : GT)
        Regular expression of donor site
    acceptor_site_pattern : str (default : AG)
        Regular expression of acceptor site

    Returns:
    ----------
    list (tuple)
        List of guessed intron boundarys
    """
    donor_site_pattern = donor_site_pattern or 'GT'
    acceptor_site_pattern = acceptor_site_pattern or 'AG'
    ann_donor_sites = [site + 1 for site in find_substr(donor_site_pattern,seq)]
    ann_acceptor_sites = [site + 1 for site in find_substr(acceptor_site_pattern,seq,False)]
    type_ =  {}
    for site in ann_donor_sites:
        type_[site]='D'
    for site in ann_acceptor_sites:
        type_[site]='A'
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
                    boundarys.append((previous_site,site))
                    previous_site = None
    return boundarys

def guess_ann(chrom,strand,length,seq,gff,donor_site_pattern=None,acceptor_site_pattern=None):
    """Get guessed AnnSequence based on donor site pattern and acceptor site pattern
        
    Parameters:
    ----------
    chrom : str
        Chromosome id to be chosen
    length : int
        Length of chromosome
    strand : str
        Strand of chromosome
    gff : pd.DataFrame    
        GFF data about RNAs and introns
    seq : str
        DNA sequence which its direction is 5' to 3'
    donor_site_pattern : str (default : GT)
        Regular expression of donor site
    acceptor_site_pattern : str (default : AG)
        Regular expression of acceptor site

    Returns:
    ----------
    AnnSequence
        Guessed AnnSequence based on donor site pattern and acceptor site pattern
    """
    selected_gff = gff[gff['chr']==chrom]
    rnas = selected_gff[selected_gff['feature'].isin(RNA_TYPES)].to_dict('record')
    ANN_TYPES = ['exon','intron','other']
    ann_seq = AnnSequence(ANN_TYPES,length)
    ann_seq.strand = strand
    ann_seq.id = ann_seq.chromosome_id = chrom
    for rna in rnas:
        rna_boundary = rna['start'],rna['end']
        start = rna['start']
        end = rna['end']
        subseq = seq[start-1:end]
        intron_boundarys = guess_boundarys(subseq,donor_site_pattern=donor_site_pattern,
                                           acceptor_site_pattern=acceptor_site_pattern)
        intron_boundarys = [(start+site[0]-1,start+site[1]-1) for site in intron_boundarys]
        intron_boundarys = get_fixed_intron_boundary(rna_boundary,intron_boundarys)
        exon_boundarys = get_exon_boundary(rna_boundary,intron_boundarys)
        for boundary in intron_boundarys:
            ann_seq.add_ann('intron',1,boundary[0]-1,boundary[1]-1)
        for boundary in exon_boundarys:
            ann_seq.add_ann('exon',1,boundary[0]-1,boundary[1]-1)

    other = get_background(ann_seq,['exon','intron'])
    ann_seq.add_ann('other',other)
    return ann_seq
