import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import write_gff
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser
from sequence_annotation.genome_handler.ann_seq_converter import GeneticBedSeqConverter
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.ann_seq_processor import get_mixed_seq
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer

class SingleSiteException(Exception):
    pass

def _count_occurrence(sites):
    count = {}
    for site in sites:
        if site not in count:
            count[site] = 0
        count[site] += 1
    return count

def _add_to_set(dict_,key,value):
    if key not in dict_.keys():
        dict_[key] = set()
    dict_[key].add(value)

def _get_id(start,end):
    return "{}_{}".format(start,end)
 
def _get_noncanonical_region(seqs,canonical_regions):
    converter = GeneticBedSeqConverter()
    region_extractor = RegionExtractor()
    regions = SeqInfoContainer()
    ann_seqs = [converter.convert(item) for item in seqs]
    mixed_ann_seq = ann_seqs[0].copy()
    for seq in ann_seqs[1:]:
        for type_ in seq.ANN_TYPES:
            mixed_ann_seq.add_ann(type_,seq.get_ann(type_))
    mixed_ann_seq = get_mixed_seq(mixed_ann_seq)
    mixed_regions = []
    for region in region_extractor.extract(mixed_ann_seq):
        if region.ann_type == 'exon_intron':
            mixed_regions.append(region)

    noncanonical_regions = {}
    for id_,type_ in canonical_regions.items():
        start, end = id_.split("_")
        is_mixed = False
        for region in mixed_regions:
            if region.start <= int(start) and int(end) <= (region.end+1):
                is_mixed = True
                break
        if is_mixed:
            region_type = None
            if type_ == 'exon':
                region_type = 'exon_skipping'
            elif type_ == 'intron':
                region_type = 'intron_retetion'
            if region_type is not None:
                noncanonical_regions[id_] = region_type
    #return zero-site based site
    return noncanonical_regions
    
def _get_region_list(regions):
    region_list = []
    for id_,type_ in regions.items():
        start,end = id_.split("_")
        region_list.append({'start':int(start),'end':int(end),'type':type_})
    return region_list
    
def _validate_canonical_path(canonical_region_list,length):
    #Valid canonical region
    canonical_path = {}
    for canonical_region in canonical_region_list:
        start,end = canonical_region['start'], canonical_region['end']
        canonical_path[start] = end
    start = 0
    is_valid = True
    while start != length:
        if start not in canonical_path.keys():
            is_valid = False
            break
        start = canonical_path[start]
    if not is_valid:
        raise Exception("Canonical path is not complete",canonical_path)
    
def _get_canonical_and_noncanonical_region_list(seqs,canonical_regions,length):
    noncanonical_regions = _get_noncanonical_region(seqs,canonical_regions)
    canonical_region_list = _get_region_list(canonical_regions)
    noncanonical_region_list = _get_region_list(noncanonical_regions)
    #Valid canonical region
    _validate_canonical_path(canonical_region_list,length)
    #return zero-site based site
    return canonical_region_list,noncanonical_region_list
    
def _get_canonical_region_and_splice_site(seqs,start_sites,end_sites,select_site_by_election=False):
    if len(start_sites) != 1:
        raise SingleSiteException("Start sites should be same in each cluster, but get {}".format(start_sites))

    if len(end_sites) != 1:
        raise SingleSiteException("End sites should be same in each cluster, but get {}".format(end_sites))

    strand = seqs[0]['strand']
    donor_sites = []
    acceptor_sites = []
    for seq in seqs:
        if strand == 'plus':
            #Convert zero-nt-based coordinate to zero-site-based coordinate
            donor_sites += [site+1 for site in seq['block_related_end']]
            acceptor_sites += seq['block_related_start']
        else:
            #Convert zero-nt-based coordinate to zero-site-based coordinate
            acceptor_sites += [site+1 for site in seq['block_related_end']]
            donor_sites += seq['block_related_start']

    sites = set()
    site_type = {}
    site_count = {}
    for donor_site,acceptor_site in zip(donor_sites,acceptor_sites):
        _add_to_set(site_type,donor_site,'D')
        _add_to_set(site_type,acceptor_site,'A')
        sites.add(donor_site)
        sites.add(acceptor_site)
        if donor_site not in site_count:
            site_count[donor_site] = 0
        if acceptor_site not in site_count:
            site_count[acceptor_site] = 0 
        site_count[donor_site] += 1
        site_count[acceptor_site] += 1
            
    sites = sorted(list(sites))
    #Linker to other altenative site
    alt_donor = {}
    alt_acceptor = {}
    #Group of altenative sites
    alt_donors = {}
    alt_acceptors = {}
    canonical_regions = {}
    donor_acceptor_site = set()
    canonical_accepts = set()
    canonical_donors = set()
    if strand == 'plus':
        range_ = range(1,len(sites))
    else:
        range_ = range(len(sites)-1,0,-1)
    for index in range_:
        if strand == 'plus':
            start = sites[index-1]
            end = sites[index]
        else:
            end = sites[index-1]
            start = sites[index]
        start_types = site_type[start]
        end_types = site_type[end]
        if start <= end:
            id_ = _get_id(start,end)
        else:
            id_ = _get_id(end,start)

        if len(start_types) != 1 and len(end_types) != 1:
            raise Exception()
        if len(start_types) != 1:
            donor_acceptor_site.add(start)
        if len(end_types) != 1:
            donor_acceptor_site.add(end)

        if 'A' in start_types and 'D' in end_types:
            canonical_regions[id_] = 'exon'
            canonical_accepts.add(start)
            canonical_donors.add(end)
        elif 'D' in start_types and 'A' in end_types:
            canonical_regions[id_] = 'intron'
            canonical_donors.add(start)
            canonical_accepts.add(end)
        if 'D' in start_types and 'D' in end_types:
            if start in alt_donor.keys():
                start = alt_donor[start]
            alt_donor[end] = start
            _add_to_set(alt_donors,start,end)
            canonical_regions[id_] = 'alt_donor'

        elif 'A' in start_types and 'A' in end_types:
            if start in alt_acceptor.keys():
                start = alt_acceptor[start]
            alt_acceptor[end] = start
            _add_to_set(alt_acceptors,start,end)
            canonical_regions[id_] = 'alt_acceptor'

    if select_site_by_election:
        for alt_sites,canonical_sites in zip([alt_donors,alt_acceptors],[canonical_donors,canonical_accepts]):
            for first_site,alts in alt_sites.items():
                sites = [first_site]+alts
                noncanonical_sites = sites - canonical_sites
                canonical_site = sites - noncanonical_sites
                noncanonical_count = [site_count[site] for site in noncanonical_sites]
                max_count = max(noncanonical_count)
                if site_count[canonical_site] < max_count:
                    end = noncanonical_count[noncanonical_count.index(max_count)]
                    id_ = _get_id(first_site,end)
                    canonical_regions[id_] = 'exon'

    #print("Parsing for canonical region and non-canonical region")
    #Merge regions with same type
    for index in range(1,len(sites)-1):
        previous = sites[index-1]
        current = sites[index]
        next_ = sites[index+1]
        previous_id = _get_id(previous,current)
        current_id = _get_id(current,next_)
        if previous_id in canonical_regions.keys() and current_id in canonical_regions.keys():
            previous_type = canonical_regions[previous_id]
            current_type = canonical_regions[current_id]
            if previous_type == current_type:
                canonical_regions[_get_id(previous,next_)] = previous_type
                del canonical_regions[_get_id(previous,current)]
                del canonical_regions[_get_id(current,next_)]

    #return zero-site based site
    #print(alt_donor,alt_acceptor)
    return canonical_regions,alt_donor,alt_acceptor,alt_donors,alt_acceptors,donor_acceptor_site

def parse(bed_path,relation_path,select_site_by_election=False):
    """Return site-based data"""
    #Read bed file form path
    parser = BedInfoParser()
    mRNAs = parser.parse(bed_path)
    parents = pd.read_csv(relation_path,sep='\t').to_dict('list')
    parents = dict(zip(parents['transcript_id'],parents['gene_id']))
    #Cluster mRNAs to genes
    genes = {}
    for mRNA in mRNAs:
        parent = parents[mRNA['id']]
        if parent not in genes.keys():
            genes[parent] = []
        genes[parent].append(mRNA)

    #Handle each cluster
    SITE_NAMES = ['canonical_regions','noncanonical_regions','alt_donor','alt_acceptor',
                  'alt_donors','alt_acceptors','donor_acceptor_site']
    GLOBAL_NAMES = ['strand','chr','start','end','score']
    gene_info = []

    for parent,mRNAs in genes.items():
        #print(parent)
        try:
            parsed = None
            start_sites = list(int(mRNA['start']) for mRNA in mRNAs)
            end_sites = list(int(mRNA['end']) for mRNA in mRNAs)
            if select_site_by_election:
                strand = mRNAs[0]['strand']
                start_count = _count_occurrence(start_sites)
                max_start_count = max(start_count.values())
                start_sites = []
                for site,count in start_count.items():
                    if max_start_count == count:
                        start_sites.append(site)
                if strand == 'plus':
                    start_sites = [min(start_sites)]
                else:
                    start_sites = [max(start_sites)]
                mRNAs_ = []
                for mRNA in mRNAs:
                    if mRNA['start']==start_sites[0]:
                        mRNAs_.append(mRNA)
                mRNAs = mRNAs_
                end_sites = list(int(mRNA['end']) for mRNA in mRNAs)
                end_count = _count_occurrence(end_sites)
                max_end_count = max(end_count.values())
                end_sites = []
                for site,count in end_count.items():
                    if max_end_count == count:
                        end_sites.append(site)
                if strand == 'plus':
                    end_sites = [max(end_sites)]
                else:
                    end_sites = [min(end_sites)]
                mRNAs_ = []
                for mRNA in mRNAs:
                    if mRNA['start']==start_sites[0] and mRNA['end']==end_sites[0]:
                        mRNAs_.append(mRNA)
                mRNAs = mRNAs_
            start_sites = list(set(start_sites))
            end_sites = list(set(end_sites))
            if len(mRNAs) > 0:
                parsed = _get_canonical_region_and_splice_site(mRNAs,start_sites,end_sites)
                canonical_regions,alt_d,alt_a,alt_ds,alt_as,donor_acceptor_site = parsed
                length = end_sites[0] - start_sites[0] + 1
                c_regions,nc_regions = _get_canonical_and_noncanonical_region_list(mRNAs,canonical_regions,length)
                parsed = c_regions,nc_regions,alt_d,alt_a,alt_ds,alt_as,donor_acceptor_site
            else:
                print("Cannot get {}'s canonical gene model".format(parent))
        except:
            raise Exception(mRNAs)

        if parsed is not None:
            data = {}
            for index,name in enumerate(SITE_NAMES):
                data[name] = parsed[index]

            for name in GLOBAL_NAMES:
                data[name] = mRNAs[0][name]

            data['id'] = parent
            gene_info.append(data)

    return gene_info

def _to_gff_item(item):
    regions = []
    strand = '+' if item['strand']=='plus' else '-'
    block = {'chr':item['chr'],'strand':strand,
             'source':'.','frame':'.','id':'.','score':item['score']}
    gene = dict(block)
    #Convert zero-nt based to one-nt based
    gene['start'] = item['start'] + 1
    gene['end'] = item['end'] + 1
    gene['feature'] = 'gene'
    gene['attribute'] = 'ID={}'.format(item['id'])
    regions.append(gene)
    for info in item['canonical_regions']+item['noncanonical_regions']:
        region = dict(block)
        region['start'] = info['start'] + gene['start']
        region['end'] = info['end'] + gene['start'] - 1
        region['feature'] = info['type']
        region['attribute'] = "Parent={}".format(item['id'])
        regions.append(region)

    alt_acceptor_sites = list(item['alt_acceptor'].values())
    alt_acceptor_sites += list(item['alt_acceptor'].keys())
    alt_donor_sites = list(item['alt_donor'].values())
    alt_donor_sites += list(item['alt_donor'].keys())
    alt_acceptor_sites = set(alt_acceptor_sites)
    alt_donor_sites = set(alt_donor_sites)
    
    for site in alt_acceptor_sites:
        alt_site = dict(block)
        alt_site['feature'] = 'alt_acceptor_site'
        alt_site['end'] = site + gene['start']
        alt_site['start'] = alt_site['end'] - 1
        alt_site['attribute'] = "Parent={}".format(item['id'])
        regions.append(alt_site)
        
    for site in alt_donor_sites:
        alt_site = dict(block)
        alt_site['feature'] = 'alt_donor_site'
        alt_site['end'] = site + gene['start']
        alt_site['start'] = alt_site['end'] - 1
        alt_site['attribute'] = "Parent={}".format(item['id'])
        regions.append(alt_site)
        
    return regions    

def parsed_data_to_gff(data):
    regions = []
    for item in data:
        regions += _to_gff_item(item)
    gff = pd.DataFrame.from_dict(regions)
    return gff

def alt_event_count(paths):
    count = {'alt_donor':0,'alt_acceptor':0,'donor_acceptor_site':0}
    for path_data in paths:
        noncanonical_regions = path_data['noncanonical_regions']
        alt_donor = path_data['alt_donor']
        alt_acceptor = path_data['alt_acceptor']
        donor_acceptor_site = path_data['donor_acceptor_site']
        for noncanonical_region in noncanonical_regions:
            type_ = noncanonical_region['type']
            if type_ not in count.keys():
                count[type_] = 0
            count[type_] += 1
        count['alt_donor'] += len(alt_donor)
        count['alt_acceptor'] += len(alt_acceptor)
        count['donor_acceptor_site'] += len(donor_acceptor_site)
        
    return count

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--bed_path",required=True)
    parser.add_argument("-t", "--id_table_path",required=True)
    parser.add_argument("-o", "--gff_path",required=True)
    parser.add_argument("--select_site_by_election",action='store_true')
    args = parser.parse_args()
    parsed = parse(args.bed_path,args.id_table_path,
                  select_site_by_election=args.select_site_by_election)
    gff=parsed_data_to_gff(parsed)
    write_gff(gff,args.gff_path)
