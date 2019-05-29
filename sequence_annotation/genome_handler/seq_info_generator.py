from .sequence import SeqInformation
from .seq_container import SeqInfoContainer
from ..utils.exception import NegativeNumberException

def _group_by_type(seqs):
    tree = {}
    for seq in seqs:
        key = seq.ann_type
        if key not in tree.keys():
            tree[key] = []
        tree[key].append(seq)
    return tree

def _remove_end_regions(regions,chroms_info,max_length):
    clean_regions = []
    not_used_regions = []
    for item in regions:
        chrom_length = chroms_info[item.chromosome_id]
        if not (item.start < max_length or item.end < max_length or
                item.start >= chrom_length-max_length or
                item.end >= chrom_length-max_length):
            clean_regions.append(item)
        else:
            not_used_regions.append(item)
    if len(not_used_regions) > 0:
        print("Following regions won't be used to generate data:")
        for item in not_used_regions:
            print(item.id)
    return clean_regions

class SeqInfoGenerator:
    """Extract selected regions' annotation information"""
    def __init__(self):
        self._seed_id = 0
        self._seq_id = 0
        self.seq_id_prefix = 'seq'
        self.seed_id_prefix = 'seed'

    def _init_id(self):
        self._seed_id = 0
        self._seq_id = 0

    def generate(self, region_container, chroms_info, half_length,modes):
        self._init_id()
        if region_container.is_empty():
            raise Exception("Container size sould larger than 0")
        seqs_info = SeqInfoContainer()
        selected_region_list = []
        regions = _group_by_type(region_container)
        for region in regions.values():
            temp= _remove_end_regions(region,chroms_info,half_length)
            selected_region_list += temp
        seeds = self._create_seeds(selected_region_list,modes)
        for seed in seeds:
            length = chroms_info[seed.chromosome_id]
            seqs_info.add(self._create_seq_info(seed,length,half_length))
        return seeds,seqs_info

    def _create_seed(self, region, mode):
        """Create seed from region"""
        if mode == 'middle':
            center = int((region.start+region.end)/2)
        elif mode == 'five_prime':
            if region.strand=='plus':
                center = region.start
            elif region.strand =='minus':
                center = region.end
            else:
                raise Exception("Strand type is not correct")
        elif mode == 'three_prime':
            if region.strand == 'plus':
                center = region.end
            elif region.strand == 'minus':
                center = region.start
            else:
                raise Exception("Strand type is not correct")
        else:
            raise Exception(mode+" is not implement yet")
        seed = SeqInformation().from_dict(region.to_dict())
        seed.source = region.id
        seed.id =  self.seed_id_prefix + "_" + str(self._seed_id)
        seed.ann_type = region.ann_type
        seed.extra_index_name = mode
        seed.extra_index = center
        self._seed_id += 1
        return seed

    def _create_seq_info(self, seed,length,half_length):
        sequence_info = SeqInformation().from_dict(seed.to_dict())
        sequence_info.source = seed.source + "_" + seed.id
        index = sequence_info.extra_index
        new_start = index - half_length
        new_end = index + half_length
        if new_end >=  length:
            raise Exception("index out of range")
        if new_start < 0:
            raise Exception("index out of range")
        sequence_info.start = new_start
        sequence_info.end = new_end
        sequence_info.id = self.seq_id_prefix + "_" + str(self._seq_id)
        self._seq_id += 1
        return sequence_info

    def _create_seeds(self, region_list, modes):
        """Create seeds from regions"""
        seeds = SeqInfoContainer()
        for region in region_list:
            for mode in modes:
                seed = self._create_seed(region, mode)
                seeds.add(seed)
        return seeds
