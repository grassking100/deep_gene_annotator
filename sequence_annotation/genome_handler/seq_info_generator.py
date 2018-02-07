import numpy as np
import random
from . import SeqInformation
from . import SeqInfoContainer
from . import DictValidator
from . import ReturnNoneException
from . import validate_return
class SeqInfoGenerator:
    """Extract selected regions' annotation information"""
    def __init__(self, region_container, principle, chroms_info,
                 seed_id_prefix, seq_id_prefix):
        self._region_container = region_container
        self._principle = principle
        self._chroms_info = chroms_info
        self._seeds = None
        self._seqs_info = None
        self._seed_id = 0
        self._seq_id = 0
        self._seed_id_prefix = seed_id_prefix
        self._seq_id_prefix = seq_id_prefix
    @property
    @validate_return("use method generate before access the data")
    def seqs_info(self):
        return self._seqs_info
    @property
    @validate_return("use method generate before access the data")
    def seeds(self):
        return self._seeds
    @property
    def valid_data_keys(self):
        return ['remove_end_of_strand','with_random_choose',
                'replaceable','each_region_number',
                'sample_number_per_region','half_length','max_diff']
    def _validate(self):
        """Validate required data"""
        dict_validator = DictValidator(self._principle)
        dict_validator.key_must_included = self.valid_data_keys
        dict_validator.validate()
    def generate(self):
        self._validate()
        self._seqs_info = SeqInfoContainer()
        selected_region_list = []
        regions = self._group_by_type(self._region_container)
        remove_end_of_strand = self._principle['remove_end_of_strand']
        for type_,number in self._principle['each_region_number'].items():
            if remove_end_of_strand:
                clean_regions= self._remove_end_regions(regions[type_])
            else:
                clean_regions = regions[type_]
            selected_region_list += self._selected_region_list(clean_regions, number)
        self._seeds = self._create_seeds(selected_region_list)
        for seed in self._seeds.data:
            seqs = self._create_seqs_info(seed).data
            for seq in seqs:
                self._seqs_info.add(seq)
    def _group_by_type(self, seqs_info):
        tree = {}
        for seq_info in seqs_info.data:
            key = seq_info.ann_type
            if key not in tree.keys():
                tree[key] = []
            tree[key].append(seq_info)
        return tree
    def _selected_region_list(self,regions,number):
        """Selected regions based on principle"""
        replaceable = self._principle['replaceable']
        with_random_choose = self._principle['with_random_choose']
        if not with_random_choose:
            region_list = regions
        else:
            region_list = list(np.random.choice(regions,number,replace=replaceable))
        return region_list
    def _remove_end_regions(self, regions):
        clean_regions = []
        not_used_regions = []
        for item in regions:
            chrom_length = self._chroms_info[item.chromosome_id]
            if not (item.start == 0 or item.end == 0 or
                    item.start == chrom_length-1 or
                    item.end == chrom_length-1):
                clean_regions.append(item)
            else:
                not_used_regions.append(item)
        if len(not_used_regions)>0:
            print("Following regions won't be used to generate data:")
            for item in not_used_regions:
                print(item.to_dict())
        return clean_regions
    @property
    def mode_text(self):
        return ['start', 'end', 'middle']
    def _create_seed(self, region):
        """Create seed from region"""
        mode = [region.start, region.end, 
              int((region.start+region.end)/2)]
        mode_selected_index = random.randint(0, len(mode)-1)
        seed = SeqInformation(region)
        seed.source = region.source+"_"+region.id
        seed.id = self._seed_id_prefix + "_" + str(self._seed_id)
        seed.type_ = region.ann_type
        seed.ann_status = self.mode_text[mode_selected_index]
        seed.extra_index_name = self.mode_text[mode_selected_index]
        seed.extra_index = mode[mode_selected_index]
        self._seed_id += 1
        return seed
    def _create_seq_info(self, seed):
        length = self._chroms_info[seed.chromosome_id]
        sequence_info = SeqInformation(seed)
        sequence_info.source = seed.source+"_"+seed.id
        half_length = self._principle['half_length']
        max_diff = self._principle['max_diff']
        index = sequence_info.extra_index
        new_start = index-(half_length+random.randint(0, max_diff))
        new_end = index+(half_length+random.randint(0, max_diff))
        if new_start >=  length:
            new_start = length - 1
        if new_end < 0:
            new_end = 0
        sequence_info.start = new_start
        sequence_info.end = new_end
        sequence_info.id = self._seq_id_prefix+"_"+str(self._seq_id)
        self._seq_id += 1
        return sequence_info
    def _create_seeds(self, region_list):
        """Create seeds from regions"""
        seeds = SeqInfoContainer()
        for region in region_list:
            seed = self._create_seed(region)
            seeds.add(seed)
        return seeds
    def _create_seqs_info(self, seed):
        sequences_info = SeqInfoContainer()
        number = self._principle['sample_number_per_region']
        for index in range(number):
            temp = self._create_seq_info(seed)
            sequences_info.add(temp)
        return sequences_info
