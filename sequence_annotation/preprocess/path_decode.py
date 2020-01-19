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

def add_to_set(dict_,key,value):
    if key not in dict_.keys():
        dict_[key] = set()
    dict_[key].add(value)

def find_path(seqs,start_sites,end_sites,select_site_by_election=False):
    def get_id(start,end):
        return "{}_{}".format(start,end)

    if len(start_sites) != 1:
        raise SingleSiteException("Start sites should be same in each cluster, but get {}".format(start_sites))

    if len(end_sites) != 1:
        raise SingleSiteException("End sites should be same in each cluster, but get {}".format(end_sites))

    length = end_sites[0] - start_sites[0] + 1
    strand = seqs[0]['strand']
    converter = GeneticBedSeqConverter()
    region_extractor = RegionExtractor()
    regions = SeqInfoContainer()
    ann_seqs = [converter.convert(item) for item in seqs]
    site_count = {}
    for seq in ann_seqs:
        regions_ = region_extractor.extract(seq)
        regions.add(regions_)
        for region in regions_:
            start = region.start
            end = region.end + 1
            if start not in site_count:
                site_count[start] = 0
            if end not in site_count:
                site_count[end] = 0 
            site_count[start] += 1
            site_count[end] += 1
                
    mixed_ann_seq = ann_seqs[0].copy()
    for seq in ann_seqs[1:]:
        for type_ in seq.ANN_TYPES:
            mixed_ann_seq.add_ann(type_,seq.get_ann(type_))

    mixed_ann_seq = get_mixed_seq(mixed_ann_seq)
    mixed_regions = []
    for region in region_extractor.extract(mixed_ann_seq):
        if region.ann_type == 'exon_intron':
            mixed_regions.append(region)

    sites = set()
    site_type = {}
    for seq in regions:
        if seq.ann_type == 'exon':
            if strand == 'plus':
                start_type = 'A'
                end_type = 'D'
            else:
                start_type = 'D'
                end_type = 'A'
        else:
            if strand == 'plus':
                start_type = 'D'
                end_type = 'A'
            else:
                start_type = 'A'
                end_type = 'D'
        #Convert zero-nt-based coordinate to zero-site-based coordinate
        start = seq.start
        end = seq.end + 1
        add_to_set(site_type,start,start_type)
        add_to_set(site_type,end,end_type)
        sites.add(start)
        sites.add(end)

    sites = sorted(list(sites))
    #Linker to other altenative donor site
    alt_donor = {}
    #Linker to other altenative accept site
    alt_accept = {}
    #Group of altenative donor sites
    alt_donors = {}
    #Group of altenative accept sites
    alt_accepts = {}
    canonical_region = {}
    donor_accept_site = set()
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
            id_ = get_id(start,end)
        else:
            id_ = get_id(end,start)

        if len(start_types) != 1 and len(end_types) != 1:
            raise Exception()
        if len(start_types) != 1:
            donor_accept_site.add(start)
        if len(end_types) != 1:
            donor_accept_site.add(end)

        if 'A' in start_types and 'D' in end_types:
            canonical_region[id_] = 'exon'
            canonical_accepts.add(start)
            canonical_donors.add(end)
        elif 'D' in start_types and 'A' in end_types:
            canonical_region[id_] = 'intron'
            canonical_donors.add(start)
            canonical_accepts.add(end)
        if 'D' in start_types and 'D' in end_types:
            if start in alt_donor.keys():
                start = alt_donor[start]
            alt_donor[end] = start
            add_to_set(alt_donors,start,end)
            canonical_region[id_] = 'alt_donor'
        elif 'A' in start_types and 'A' in end_types:
            if start in alt_accept.keys():
                start = alt_accept[start]
            alt_accept[end] = start
            add_to_set(alt_accepts,start,end)
            canonical_region[id_] = 'alt_accept'
    
    if select_site_by_election:
        for alt_sites,canonical_sites in zip([alt_donors,alt_accepts],[canonical_donors,canonical_accepts]):
            for first_site,alts in alt_sites.items():
                sites = [first_site]+alts
                noncanonical_sites = sites - canonical_sites
                canonical_site = sites - noncanonical_sites
                noncanonical_count = [site_count[site] for site in noncanonical_sites]
                max_count = max(noncanonical_count)
                if site_count[canonical_site] < max_count:
                    end = noncanonical_count[noncanonical_count.index(max_count)]
                    id_ = get_id(first_site,end)
                    print(id_)
                    canonical_region[id_] = 'exon'

    #Merge regions with same type
    for index in range(1,len(sites)-1):
        previous = sites[index-1]
        current = sites[index]
        next_ = sites[index+1]
        previous_id = get_id(previous,current)
        current_id = get_id(current,next_)
        if previous_id in canonical_region.keys() and current_id in canonical_region.keys():
            previous_type = canonical_region[previous_id]
            current_type = canonical_region[current_id]
            if previous_type == current_type:
                canonical_region[get_id(previous,next_)] = previous_type
                del canonical_region[get_id(previous,current)]
                del canonical_region[get_id(current,next_)]

    noncanonical_region = {}
    for id_,type_ in canonical_region.items():
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
                noncanonical_region[id_] = region_type

    canonical_region_list = []
    noncanonical_region_list = []
    for id_,type_ in canonical_region.items():
        start,end = id_.split("_")
        canonical_region_list.append({'start':int(start),'end':int(end),'type':type_})
    for id_,type_ in noncanonical_region.items():
        start,end = id_.split("_")
        noncanonical_region_list.append({'start':int(start),'end':int(end),'type':type_})

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
    #return zero site based site
    return canonical_region_list,noncanonical_region_list,alt_donor,alt_accept,alt_donors,alt_accepts,donor_accept_site

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
    path_names = ['canonical_regions','noncanonical_regions','alt_donor','alt_accept',
                  'alt_donors','alt_accepts','donor_accept_site']
    global_names = ['strand','chr','start','end','score']
    gene_info = []
    def count_occurrence(sites):
        count = {}
        for site in sites:
            if site not in count:
                count[site] = 0
            count[site] += 1
        return count
    
    for parent,mRNAs in genes.items():
        try:
            parsed = None
            start_sites = list(int(mRNA['start']) for mRNA in mRNAs)
            end_sites = list(int(mRNA['end']) for mRNA in mRNAs)
            if select_site_by_election:
                strand = mRNAs[0]['strand']
                start_count = count_occurrence(start_sites)
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
                end_count = count_occurrence(end_sites)
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
                parsed = find_path(mRNAs,start_sites,end_sites)
            else:
                print("Cannot get {}'s canonical gene model".format(parent))
        except:
            raise Exception(mRNAs)

        if parsed is not None:
            data = {}
            for index,name in enumerate(path_names):
                data[name] = parsed[index]

            for name in global_names:
                data[name] = mRNAs[0][name]

            data['id'] = parent
            #Convert zero-nt based to zero-site based
            data['end'] += 1
            gene_info.append(data)

    return gene_info

def alt_event_count(paths):
    count = {'alt_donor':0,'alt_accept':0,'donor_accept_site':0}
    for path_data in paths:
        noncanonical_regions = path_data['noncanonical_regions']
        alt_donor = path_data['alt_donor']
        alt_accept = path_data['alt_accept']
        donor_accept_site = path_data['donor_accept_site']
        for noncanonical_region in noncanonical_regions:
            type_ = noncanonical_region['type']
            if type_ not in count.keys():
                count[type_] = 0
            count[type_] += 1
        count['alt_donor'] += len(alt_donor)
        count['alt_accept'] += len(alt_accept)
        count['donor_accept_site'] += len(donor_accept_site)
        
    return count

def get_tss(parsed_data):
    sites = {}
    for parsed in parsed_data:
        id_ = "{}_{}".format(parsed['chr'],parsed['strand'])
        if id_ not in sites.keys():
            sites[id_] = set()
        if parsed['strand'] == 'plus':    
            sites[id_].add(parsed['start'])
        else:   
            sites[id_].add(parsed['end'])
    return sites
    
def get_cleavage_site(parsed_data):
    sites = {}
    for parsed in parsed_data:
        id_ = "{}_{}".format(parsed['chr'],parsed['strand'])
        if id_ not in sites.keys():
            sites[id_] = set()
        if parsed['strand'] == 'plus':
            sites[id_].add(parsed['end'])
        else:   
            sites[id_].add(parsed['start'])
    return sites
    
def get_donor_site(parsed_data):
    sites = {}
    alt_sites = {}
    for parsed in parsed_data:
        id_ = "{}_{}".format(parsed['chr'],parsed['strand'])
        if id_ not in sites.keys():
            sites[id_] = set()
            alt_sites[id_] =  set()
        canonical_regions = parsed['canonical_regions']
        alt_donors = parsed['alt_donors']
        start = parsed['start']
        for region in canonical_regions:
            if parsed['strand'] == 'plus':
                site = region['start']
            else:
                site = region['end']  
            if region['type']=='intron':    
                sites[id_].add(site+start)
                if site in alt_donors.keys():
                    alt_sites[id_].add(site+start)
                    alt_sites[id_] = alt_sites[id_].union(set(site+start for site in alt_donors[site]))
        
    return sites,alt_sites
    
def get_accept_site(parsed_data):
    sites = {}
    alt_sites = {}
    for parsed in parsed_data:
        id_ = "{}_{}".format(parsed['chr'],parsed['strand'])
        if id_ not in sites.keys():
            sites[id_] = set()
            alt_sites[id_] =  set()
        canonical_regions = parsed['canonical_regions']
        alt_accepts = parsed['alt_accepts']
        start = parsed['start']
        for region in canonical_regions:
            if parsed['strand'] == 'plus':
                site = region['end']
            else:
                site = region['start']
            if region['type']=='intron':    
                sites[id_].add(site+start)
                
                if site in alt_accepts.keys():
                    alt_sites[id_].add(site+start)
                    alt_sites[id_] = alt_sites[id_].union(set(site+start for site in alt_accepts[site]))
        
    return sites,alt_sites

def _get_safe_sites(sites,alt_sites=None,dist=None):
    dist = dist or 32
    valid_sites = set()
    sites = sorted(list(sites))
    valid_sites = set()
    for index in range(len(sites)):
        is_valid = True
        current_site = sites[index]
        if index < len(sites)-1:
            next_site = sites[index+1]
            if abs(current_site-next_site) <= dist:
                is_valid = False
        if index > 0:
            previous_site = sites[index-1]
            if abs(current_site-previous_site) <= dist:
                is_valid = False
        if is_valid:        
            valid_sites.add(current_site)    
    if alt_sites:        
        valid_sites -= alt_sites
    return valid_sites

def get_safe_sites(sites,alt_sites=None,dist=None):
    valid_sites = {}
    for id_ in sites.keys():
        sites_ = sites[id_]
        alt_sites_ = None
        if alt_sites:        
            alt_sites_ = alt_sites[id_]
        valid_sites_ = _get_safe_sites(sites_,alt_sites_,dist)
        valid_sites[id_] = valid_sites_
    return valid_sites

def _to_gff_item(item):
    regions = []
    strand = '+' if item['strand']=='plus' else '-'
    block = {'chr':item['chr'],'strand':strand,
             'source':'.','frame':'.','id':'.','score':item['score']}
    gene = dict(block)
    gene['start'] = item['start'] + 1
    gene['end'] = item['end']
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

    alt_accept_sites = list(item['alt_accept'].values())
    alt_accept_sites += list(item['alt_accept'].keys())
    alt_donor_sites = list(item['alt_donor'].values())
    alt_donor_sites += list(item['alt_donor'].keys())
    alt_accept_sites = set(alt_accept_sites)
    alt_donor_sites = set(alt_donor_sites)
    
    for site in alt_accept_sites:
        alt_site = dict(block)
        alt_site['feature'] = 'alt_accept_site'
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

def alt_data_to_gff(data):
    regions = []
    for item in data:
        regions += _to_gff_item(item)
    gff = pd.DataFrame.from_dict(regions)
    return gff

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--bed_path",required=True)
    parser.add_argument("-t", "--id_table_path",required=True)
    parser.add_argument("-o", "--gff_path",required=True)
    parser.add_argument("--select_site_by_election",action='store_true')
    args = parser.parse_args()
    paths = parse(args.bed_path,args.id_table_path,
                  select_site_by_election=args.select_site_by_election)
    gff=alt_data_to_gff(paths)
    write_gff(gff,args.gff_path)
