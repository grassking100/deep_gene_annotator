import numpy as np
import random
from .sequence import SeqInformation
from .seq_container import SeqInfoContainer
from ..utils.exception import NegativeNumberException
class SeqInfoGenerator:
    """Extract selected regions' annotation information"""
    def __init__(self):
        self._seed_id = 0
        self._seq_id = 0

    def _validate_principle(self,principle):
        if principle['length_constant']:
            max_diff = principle['total_length'] - (2*principle['half_length']+1)
            if max_diff < 0:
                raise NegativeNumberException('total_length-(2*half_length+1)',
                                              max_diff)
        else:
            max_diff = principle['max_diff']
            if max_diff < 0:
                raise NegativeNumberException('max_diff',max_diff)

    def _validate(self,principle):
        """Validate required data"""
        self._validate_principle(principle)

    def generate(self, region_container, principle,
                 chroms_info, seed_id_prefix, seq_id_prefix):
        if len(region_container)==0:
            raise Exception("Container size sould larger than 0")
        self._validate(principle)
        seqs_info = SeqInfoContainer()
        selected_region_list = []
        regions = self._group_by_type(region_container,principle['each_region_number'].keys())
        remove_end_of_strand = principle['remove_end_of_strand']
        for type_,number in principle['each_region_number'].items():
            if remove_end_of_strand:
                if principle['length_constant']:
                    max_length = principle['total_length']
                else:
                    max_length = (principle['max_diff']+principle['half_length']) * 2 + 1
                temp= self._remove_end_regions(regions[type_],chroms_info,max_length)
            else:
                temp = regions[type_]
            clean_regions = SeqInfoContainer()
            clean_regions.add(temp)
            selected_region_list += list(np.random.choice(clean_regions.data,
                                                          number,
                                                          replace=principle['replaceable']))
        seeds = self._create_seeds(selected_region_list,principle['modes'],seed_id_prefix)
        number = principle['region_number_per_seed']
        for seed in seeds.data:
            length = chroms_info[seed.chromosome_id]
            seqs = self._create_seqs_info(seed,number,length,principle,seq_id_prefix).data
            for seq in seqs:
                seqs_info.add(seq)
        return seeds,seqs_info

    def _group_by_type(self, seqs_info,types=None):
        tree = {}
        if types is None:
            keys = []
            for seq_info in seqs_info:
                keys.append(seq_info.ann_type)
            types = set(keys)
        for type_ in types:
            tree[type_] = []
        for seq_info in seqs_info:
            key = seq_info.ann_type
            tree[key].append(seq_info)
        return tree

    def _remove_end_regions(self, regions,chroms_info,max_length):
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
            print("\nFollowing regions won't be used to generate data:")
            for item in not_used_regions:
                print(item.to_dict())
        return clean_regions

    def _create_seed(self, region, mode,seed_id_prefix):
        """Create seed from region"""
        if mode == 'middle':
            center = int((region.start+region.end)/2)
        elif mode == '5\'':
            if region.strand=='plus':
                center = region.start
            elif region.strand =='minus':
                center = region.end
            else:
                raise Exception()
        elif mode == '3\'':
            if region.strand == 'plus':
                center = region.end
            elif region.strand == 'minus':
                center = region.start
            else:
                raise Exception()
        else:
            raise Exception(mode+" is not implement yet")
        seed = SeqInformation().from_dict(region.to_dict())
        seed.source = region.source + "_" + region.id
        seed.id = seed_id_prefix + "_" + str(self._seed_id)
        seed.ann_type = region.ann_type
        seed.extra_index_name = mode
        seed.extra_index = center
        self._seed_id += 1
        return seed

    def _create_seq_info(self, seed,length,principle,seq_id_prefix):
        sequence_info = SeqInformation().from_dict(seed.to_dict())
        sequence_info.source = seed.source + "_" + seed.id
        half_length = principle['half_length']
        index = sequence_info.extra_index
        if principle['length_constant']:
            total_length = principle['total_length']
            max_diff = total_length - (2*half_length+1)
            diff = random.randint(0, max_diff)
            new_start = index - half_length - (max_diff - diff)
            new_end = index + half_length + diff
        else:
            max_diff = principle['max_diff']
            new_start = index - (half_length + random.randint(0, max_diff))
            new_end = index + (half_length + random.randint(0, max_diff))
        if new_end >=  length:
            raise Exception("index out of range")
        if new_start < 0:
            raise Exception("index out of range")
        sequence_info.start = new_start
        sequence_info.end = new_end
        sequence_info.id = seq_id_prefix + "_" + str(self._seq_id)
        self._seq_id += 1
        return sequence_info

    def _create_seeds(self, region_list,modes,seed_id_prefix):
        """Create seeds from regions"""
        seeds = SeqInfoContainer()
        for region in region_list:
            mode = modes[random.randint(0,len(modes)-1)]
            seed = self._create_seed(region, mode, seed_id_prefix)
            seeds.add(seed)
        return seeds

    def _create_seqs_info(self, seed,sample_number_per_region,length,principle,seq_id_prefix):
        sequences_info = SeqInfoContainer()
        for index in range(sample_number_per_region):
            temp = self._create_seq_info(seed,length,principle,seq_id_prefix)
            sequences_info.add(temp)
        return sequences_info
