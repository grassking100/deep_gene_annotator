from .seq_container import SeqInfoContainer
from .sequence import SeqInformation
from ..utils.exception import InvalidStrandType,NegativeNumberException
import random

class RegionSelector:
    def extract(self,core_regions_info,seq_length,upstream_range,chroms_length):
        selected_regions = SeqInfoContainer()
        for info in core_regions_info:
            seq_info = SeqInformation()
            seq_info.from_dict(info.to_dict())
            if seq_info.strand == 'plus':
                try:
                    seq_info.start = seq_info.extra_index - random.randint(upstream_range[0],upstream_range[1])
                    seq_info.end = seq_info.start + seq_length - 1
                    if chroms_length[str(seq_info.chromosome_id)] > seq_info.end:
                        selected_regions.add(seq_info)
                except NegativeNumberException as exp:
                    pass
            elif seq_info.strand == 'minus':
                seq_info.end = seq_info.extra_index + random.randint(upstream_range[0],upstream_range[1])
                try:
                    seq_info.start = seq_info.end - seq_length + 1
                    if chroms_length[str(seq_info.chromosome_id)] > seq_info.end:
                        selected_regions.add(seq_info)
                except NegativeNumberException as exp:
                    pass
            else:
                raise InvalidStrandType(seq_info.strand)
        return selected_regions