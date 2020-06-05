import sys
import os
import pandas as pd
import warnings
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.exception import InvalidStrandType
from sequence_annotation.utils.utils import get_gff_with_updated_attribute,read_gff,write_gff
from sequence_annotation.genome_handler.sequence import STRANDS, PLUS, MINUS
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser
from sequence_annotation.genome_handler.ann_seq_converter import GeneticBedSeqConverter
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.ann_seq_processor import get_mixed_seq
from sequence_annotation.preprocess.utils import EXON_SKIPPING,INTRON_RETENTION
from sequence_annotation.preprocess.utils import ALT_DONOR,ALT_DONOR_SITE
from sequence_annotation.preprocess.utils import ALT_ACCEPTOR,ALT_ACCEPTOR_SITE
from sequence_annotation.preprocess.get_id_table import get_id_table
from sequence_annotation.preprocess.gff2bed import gff2bed

class SingleSiteException(Exception):
    pass

EXON='exon'
INTRON='intron'
STRAND_CONVERT = {PLUS:'+',MINUS:'-'}

def _add_to_set(dict_,key,value):
    if key not in dict_.keys():
        dict_[key] = set()
    dict_[key].add(value)

def _get_id(start,end):
    if start <= end:
        id_ = "{}_{}".format(start,end)
    else:
        id_ = "{}_{}".format(end,start)
    return id_
 
def _get_noncanonical_region(seqs,canonical_regions):
    converter = GeneticBedSeqConverter()
    region_extractor = RegionExtractor()
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
            if type_ not in [EXON,INTRON]:
                raise Exception("Unknown type")
            if type_ == EXON:
                region_type = EXON_SKIPPING
            else:
                region_type = INTRON_RETENTION
            noncanonical_regions[id_] = region_type
    #return zero-site based site
    return noncanonical_regions
    
def _get_region_list(regions):
    region_list = []
    for id_,type_ in regions.items():
        if "_" in str(id_):
            start,end = id_.split("_")
        else:
            start = end = id_
        region_list.append({'start':int(start),'end':int(end),'type':type_})
    return region_list
    
def _validate_canonical_path(canonical_region_list,length,allow_partial_gene=False):
    #Partial gene could miss some of its exon at its boundary, 
    #if allow_partial_gene is True, then it ignore this error, otherwise it would raise an Exception
    #Valid canonical region
    canonical_path = {}
    for canonical_region in canonical_region_list:
        start,end = canonical_region['start'], canonical_region['end']
        canonical_path[start] = end
        
    if allow_partial_gene:
        start = min(canonical_path.keys())
        end = max(canonical_path.values())
        if 0 not in canonical_path:
            canonical_path[0] = start
        if length not in canonical_path.values():
            canonical_path[end] = length

    start = 0
    for _ in range(len(canonical_path)):
        if start not in canonical_path.keys():
            raise Exception("Canonical path {} is not complete, missing site {}",canonical_path,start)
        start = canonical_path[start]
    if start != length:
        raise Exception("Canonical path {} is not complete, missing end site {}",canonical_path,length)

def _get_canonical_and_noncanonical_region_list(seqs,canonical_regions,alt_site_regions,length,allow_partial_gene=False):
    noncanonical_regions = _get_noncanonical_region(seqs,canonical_regions)
    canonical_region_list = _get_region_list(canonical_regions)
    noncanonical_region_list = _get_region_list(noncanonical_regions) + _get_region_list(alt_site_regions)
    #Valid canonical region
    _validate_canonical_path(canonical_region_list,length,allow_partial_gene=allow_partial_gene)
    #return zero-site based site
    return canonical_region_list,noncanonical_region_list
    
def get_canonical_region_and_alt_splice(seqs,select_site_by_election=False):
    """
    Seqs should be list of dictionary which store by zero-base-based
    If select_site_by_election is False, then the most upstream site would be chosen as canoncial site. 
    Otherwise, the splicing site with the highest occurrence frequency would be chosen as 
    canonical splicing site. If there are muliple candidates, then the most upstream splicing site
    would be chosen as canoncial site
    """
    #print(seqs)
    start_sites = list(set([int(seq['start']) for seq in seqs]))
    end_sites = list(set([int(seq['end']) for seq in seqs]))
    strand = seqs[0]['strand']
    if len(start_sites) != 1:
        raise SingleSiteException("Start sites should be same in each cluster, but get {}".format(start_sites))

    if len(end_sites) != 1:
        raise SingleSiteException("End sites should be same in each cluster, but get {}".format(end_sites))

    if strand not in STRANDS:
        raise InvalidStrandType(strand)
        
    #Convert zero-nt-based coordinate to zero-site-based coordinate
    length = end_sites[0] - start_sites[0] + 1
    if strand == PLUS:
        start_site = 0
        end_site = length
    else:
        end_site = 0
        start_site = length

    donor_sites = []
    acceptor_sites = []
    site_types= {}
    site_counts = {}
    sites = []
    for seq in seqs:
        block_related_starts = list(seq['block_related_start'])
        block_related_ends = list(seq['block_related_end'])
        #Convert zero-nt-based coordinate to zero-site-based coordinate
        if strand == PLUS:
            donor_sites += [site+1 for site in block_related_ends]
            acceptor_sites += seq['block_related_start']
        else:
            acceptor_sites += [site+1 for site in block_related_ends]
            donor_sites += block_related_starts
            
    sites += acceptor_sites
    sites += donor_sites

    for removed_site in [start_site,end_site]:
        for target_sets in [donor_sites,acceptor_sites]:
            if removed_site in target_sets:
                target_sets.remove(removed_site)

    _add_to_set(site_types,start_site,'5')
    _add_to_set(site_types,end_site,'3')

    site_type_counts = {}
    for site in donor_sites:
        site_counts[site] = 0
        site_type_counts[site] = {'D':0,'A':0}

    for site in donor_sites:
        site_type_counts[site]['D']+=1
        _add_to_set(site_types,site,'D')
        site_counts[site] += 1
    
    for site in acceptor_sites:
        site_counts[site] = 0
        site_type_counts[site] = {'D':0,'A':0}

    for site in acceptor_sites:
        site_type_counts[site]['A']+=1
        _add_to_set(site_types,site,'A')
        site_counts[site] += 1
            
    sites = sorted(list(set(sites)))
    donor_acceptor_sites = set()
    #Decide forward range
    if strand == PLUS:
        range_ = range(1,len(sites))
    else:
        range_ = range(len(sites)-1,0,-1)

    #Handle ambiguous site types
    for index in range_:
        if strand == PLUS:
            current_site = sites[index-1]
            next_site = sites[index]
        else:
            next_site = sites[index-1]
            current_site = sites[index]
        current_site_types = set(site_types[current_site])
        next_site_types = set(site_types[next_site])
        for site_types_ in [current_site_types,next_site_types]:
            if '5' in site_types_:
                site_types_.remove('5')
            if '3' in site_types_:
                site_types_.remove('3')

        if len(current_site_types) > 1:
            raise Exception("The region's start site {} is {}".format(current_site,current_site_types))
            
        if len(next_site_types) > 1:
            if current_site_types == set('D'):
                site_type= 'A'
            elif current_site_types == set('A'):
                site_type= 'D'
            else:
                raise Exception("Got unknown type {} at {}".format(current_site_types,current_site))
            site_types[next_site] = set(site_type)
            warnings.warn("The end site {} is both acceptor site and donor site, the"
                          " site would be change due to its previous site type, the site would be set to {}".format(next_site,site_type))
    #Get sites by forward direction, and assign exon and intron to canonical_regions
    exon_boundary_pairs = {}
    intron_boundary_pairs = {}
    for index in range_:
        if strand == PLUS:
            start = sites[index-1]
            end = sites[index]
        else:
            end = sites[index-1]
            start = sites[index]
        start_types = site_types[start]
        end_types = site_types[end]
        id_ = _get_id(start,end)
        type_ = None
        #Single exon transcript
        if '5' in start_types and '3' in end_types:
            type_ = EXON
        #Multiple exon transcript
        elif '5' in start_types and 'D' in end_types:
            type_ = EXON
        elif '5' in start_types and 'A' in end_types:
            type_ = INTRON
        elif 'A' in start_types and '3' in end_types:
            type_ = EXON
        elif 'A' in start_types and 'D' in end_types:
            type_ = EXON     
        elif 'D' in start_types and 'A' in end_types:
            type_ = INTRON
        elif 'D' in start_types and '3' in end_types:
            type_ = INTRON
        if type_ is not None:
            if type_ == EXON:
                exon_boundary_pairs[start] = end
            else:
                intron_boundary_pairs[start] = end
                
    #Decide forward range
    if strand == PLUS:
        range_ = range(len(sites))
    else:
        range_ = range(len(sites)-1,-1,-1)
        
    #Get sites by forward direction, and assign altenative site to group
    donor_site_groups = []
    acceptor_site_groups = []
    donor_site_group = []
    acceptor_site_group = []
    for index in range_:
        site = sites[index]
        site_type = site_types[site]
        #Add site to group
        if 'D' in site_type:
            donor_site_group.append(site)
        #Add existed group to groups and create new group
        else:
            if len(donor_site_group) > 0 :
                donor_site_groups.append(donor_site_group)
            donor_site_group = []
            
        #Add site to group
        if 'A' in site_type:
            acceptor_site_group.append(site)
        #Add existed group to groups and create new group
        else:
            if len(acceptor_site_group) > 0 :
                acceptor_site_groups.append(acceptor_site_group)
            acceptor_site_group = []

    #rint(intron_boundary_pairs)
    #rint(donor_site_groups)
    #rint(acceptor_site_groups)
            
    for group in donor_site_groups:
        if len(group)>1:
            site_counts_ = [site_counts[site] for site in group] 
            exon_end = group[0]
            intron_start = group[-1]
            #Intron start and exon end
            if select_site_by_election:
                index = site_counts_.index(max(site_counts_))
                new_site = group[index]
            else:
                new_site = exon_end
            #Get exon end
            exon_boundary_starts = list(exon_boundary_pairs.keys())
            exon_boundary_ends = list(exon_boundary_pairs.values())
            exon_start = exon_boundary_starts[exon_boundary_ends.index(exon_end)]
            #Change exon end
            old_exon_end = exon_boundary_pairs[exon_start]
            if old_exon_end != new_site:
                print("Change exon end from {} to {}".format(old_exon_end,new_site))
                exon_boundary_pairs[exon_start] = new_site
            #Change intron start
            if intron_start != new_site:
                print("Change intron start from {} to {}".format(intron_start,new_site))
                intron_boundary_pairs[new_site] = intron_boundary_pairs[intron_start]
                del intron_boundary_pairs[intron_start]

    for group in acceptor_site_groups:
        if len(group)>1:
            site_counts_ = [site_counts[site] for site in group]
            intron_end = group[0]
            exon_start = group[-1]
            #Intron end and exon start
            if select_site_by_election:
                index = site_counts_.index(max(site_counts_))
                new_site = group[index]
            else:
                new_site = intron_end
            #Get intron start
            intron_boundary_starts = list(intron_boundary_pairs.keys())
            intron_boundary_ends = list(intron_boundary_pairs.values())
            intron_start = intron_boundary_starts[intron_boundary_ends.index(intron_end)]
            #Change intron end
            old_intron_end = intron_boundary_pairs[intron_start]
            if old_intron_end != new_site:
                print("Change intron end from {} to {}".format(old_intron_end,new_site))
                intron_boundary_pairs[intron_start] = new_site
            #Change exon start
            if exon_start!=new_site:
                print("Change exon start from {} to {}".format(exon_start,new_site))
                exon_boundary_pairs[new_site] = exon_boundary_pairs[exon_start]
                del exon_boundary_pairs[exon_start]
            
    #Annotated canonical regions
    canonical_regions = {}
    for start,end in exon_boundary_pairs.items():
        id_ = _get_id(start,end)
        canonical_regions[id_] = EXON
    for start,end in intron_boundary_pairs.items():
        id_ = _get_id(start,end)
        canonical_regions[id_] = INTRON
        
    canonical_accepts = set()
    canonical_donors = set()
    for start,end in intron_boundary_pairs.items():
        canonical_donors.add(start)
        canonical_accepts.add(end) 
            
    #Annotated alternative splicing regions
    alt_acceptor_sites = set()
    alt_donor_sites = set()
    alt_site_regions = {}
    for group in acceptor_site_groups:
        if len(group) > 1:
            id_ = _get_id(group[0],group[-1])
            alt_site_regions[id_] = ALT_ACCEPTOR
            alt_acceptor_sites.update(group)
            for site in group:
                if site in canonical_accepts:
                    alt_acceptor_sites.remove(site)
        
    for group in donor_site_groups:
        if len(group) > 1:
            id_ = _get_id(group[0],group[-1])
            alt_site_regions[id_] = ALT_DONOR
            alt_donor_sites.update(group)
            for site in group:
                if site in canonical_donors:
                    alt_donor_sites.remove(site)

    for start,end in canonical_regions.items():
        if start==end:
            raise Exception("Got empty length data")

    for start,end in alt_site_regions.items():
        if start==end:
            raise Exception("Got empty length data") 
            
    #Return zero-site base data
    return canonical_regions,alt_site_regions,alt_donor_sites,alt_acceptor_sites,donor_acceptor_sites,length

def _count_occurrence(sites):
    count = {}
    for site in sites:
        count[site] = 0
    for site in sites:
        count[site] += 1
    return count

def get_most_start_end_transcripts(transcripts):
    """
    Transcripts should be list of dictionary which store by zero-nt-based
    Get transcripts which have start site location and end site location with the highest frequency occurrences.
    The order of selected is start site location, and end site location.
    If there are multiple start site locations with highest frequency of occurrences, then the most upstream location would be chosen
    If there are multiple end site locations with highest frequency of occurrences, then the most upstream location would be chosen
    """
    
    strand = transcripts[0]['strand']
    if strand not in STRANDS:
        raise InvalidStrandType(strand)
    #Get transcripts which start sites have the highest frequency of occurrence
    if strand == PLUS:
        start_sites = list(int(transcript['start']) for transcript in transcripts)
    else:
        start_sites = list(int(transcript['end']) for transcript in transcripts)
    
    start_count = _count_occurrence(start_sites)
    max_start_count = max(start_count.values())
    max_start_sites = []
    for site,count in start_count.items():
        if max_start_count == count:
            max_start_sites.append(site)
    #If there are multiple start site locations with highest frequency of occurrences, then the most upstream location would be chosen
    if strand == PLUS:
        start_site = min(max_start_sites)
    else:
        start_site = max(max_start_sites)

    selected_transcripts = []
    for transcript in transcripts:
        if strand == PLUS:
            site = transcript['start']
        else:
            site = transcript['end']
        if site==start_site:
            selected_transcripts.append(transcript)

    #Get transcripts which end sites have the highest frequency of occurrence
    transcripts = selected_transcripts
    if strand == PLUS:
        end_sites = list(int(transcript['end']) for transcript in transcripts)
    else:
        end_sites = list(int(transcript['start']) for transcript in transcripts)
    end_count = _count_occurrence(end_sites)
    max_end_count = max(end_count.values())
    max_end_sites = []
    
    for site,count in end_count.items():
        if max_end_count == count:
            max_end_sites.append(site)
    #If there are multiple end site locations with highest frequency of occurrences, then the most upstream location would be chosen        
    if strand == PLUS:
        end_site = min(max_end_sites)
    else:
        end_site = max(max_end_sites)
    selected_transcripts = []

    for transcript in transcripts:
        if strand == PLUS:
            site = transcript['end']
        else:
            site = transcript['start']
        
        if site==end_site:
            selected_transcripts.append(transcript)

    return selected_transcripts

def _to_gff_item(item):
    regions = []
    strand = STRAND_CONVERT[item['strand']]
    block = {'chr':item['chr'],'strand':strand,
             'source':'.','frame':'.','id':'.','score':item['score']}
    gene = dict(block)
    #Convert zero-nt based to one-nt based
    gene['feature'] = 'gene'
    gene['id'] =  "{}_gene".format(item['id'])
    gene['start'] = item['start'] + 1
    gene['end'] = item['end'] + 1
    gene['parent'] = None
    
    transcript = dict(gene)
    transcript['feature'] = 'mRNA'
    transcript['id'] = item['id']
    transcript['parent'] = gene['id']
    
    regions.append(gene)
    regions.append(transcript)
    
    #Add zero-site based to one-nt based ,then get one-nt based data
    for info in item['canonical_regions']+item['noncanonical_regions']:
        region = dict(block)
        region['start'] = info['start'] + gene['start']
        region['end'] = info['end'] + gene['start'] - 1
        region['feature'] = info['type']
        region['parent'] = item['id']
        regions.append(region)
    
    for site in item[ALT_ACCEPTOR_SITE]:
        alt_site = dict(block)
        alt_site['feature'] = ALT_ACCEPTOR_SITE
        alt_site['end'] = site + gene['start']
        alt_site['start'] = alt_site['end'] - 1
        alt_site['parent'] = item['id']
        regions.append(alt_site)
        
    for site in item[ALT_DONOR_SITE]:
        alt_site = dict(block)
        alt_site['feature'] = ALT_DONOR_SITE
        alt_site['end'] = site + gene['start']
        alt_site['start'] = alt_site['end'] - 1
        alt_site['parent'] = item['id']
        regions.append(alt_site)
        
    return regions

def get_cluster_mRNA(bed,id_table):
    parents = id_table.to_dict('list')
    parents = dict(zip(parents['transcript_id'],parents['gene_id']))
    parser = BedInfoParser()
    #Convert to zero-base based data
    mRNAs = parser.parse(bed)
    #Cluster mRNAs to genes
    genes = {}
    for mRNA in mRNAs:
        parent = parents[mRNA['id']]
        if parent not in genes.keys():
            genes[parent] = []
        mRNA['parent'] = parent
        genes[parent].append(mRNA)
    return genes

def convert_from_bed_to_gene_with_alt_status_gff(bed,id_table,select_site_by_election=False,allow_partial_gene=False):
    """Return site-based data"""
    genes = get_cluster_mRNA(bed,id_table)
    #Handle each cluster
    gene_info = []
    for gene_id,mRNAs in genes.items():
        parsed = None
        try:
            if select_site_by_election:
                mRNAs = get_most_start_end_transcripts(mRNAs)
            parsed = get_canonical_region_and_alt_splice(mRNAs,select_site_by_election)
            canonical_regions,alt_site_regions,alt_ds,alt_as,donor_acceptor_site,length = parsed
            c_regions,nc_regions = _get_canonical_and_noncanonical_region_list(mRNAs,canonical_regions,alt_site_regions,length,
                                                                               allow_partial_gene=allow_partial_gene)
            
            parsed = c_regions,nc_regions,alt_ds,alt_as,donor_acceptor_site
            if len(donor_acceptor_site)>0:
                warnings.warn("There is at least one site being both acceptor site and donor site at {}".format(gene_id))
        except:
            raise Exception("The gene {} causes error".format(gene_id))

        SITE_NAMES = ['canonical_regions','noncanonical_regions',ALT_DONOR_SITE,ALT_ACCEPTOR_SITE,
                      'donor_acceptor_site']
        GLOBAL_NAMES = ['strand','chr','start','end']
            
        if parsed is not None:
            data = {}
            for index,name in enumerate(SITE_NAMES):
                data[name] = parsed[index]

            for name in GLOBAL_NAMES:
                #Get zero-nt data
                data[name] = mRNAs[0][name]
                
            scores = [mRNA['score'] for mRNA in mRNAs]
            if any([scores=='.' for score in scores]):
                data['score'] = '.'
            else:
                data['score'] =max(scores)

            data['id'] = gene_id
            gene_info.append(data)

    regions = []
    for item in gene_info:
        regions += _to_gff_item(item)
        
    gff = pd.DataFrame.from_dict(regions)
    gff = get_gff_with_updated_attribute(gff)    
    return gff

def convert_from_gff_to_gene_with_alt_status_gff(gff,**kwargs):
    bed = gff2bed(gff)
    id_table = get_id_table(gff)
    gff = convert_from_bed_to_gene_with_alt_status_gff(bed,id_table,**kwargs)
    return gff

def main(input_path,output_gff_path,id_table_path=None,**kwargs):
    if 'bed' in input_path.split('.')[-1]:
        if id_table_path is None:
            raise Exception("If input data is bed format, then the id_table_path must be provided")
        bed = read_bed(input_path)
        id_table = read_id_table(id_table_path)
        gene_gff = convert_from_bed_to_gene_with_alt_status_gff(bed,id_table,**kwargs)
    else:
        gff = read_gff(input_path)
        gene_gff = convert_from_gff_to_gene_with_alt_status_gff(gff,**kwargs)

    write_gff(gene_gff,output_gff_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",required=True)
    parser.add_argument("-o", "--output_gff_path",required=True)
    parser.add_argument("-t", "--id_table_path")
    parser.add_argument("--select_site_by_election",action='store_true')
    parser.add_argument("--allow_partial_gene",action='store_true')
    args = parser.parse_args()
    main(**vars(args))