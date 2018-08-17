class RegionSelector():
    def extract(self,core_regions_info,seq_length,upstream_range,chroms_length):
        selected_regions = SeqInfoContainer()
        for info in core_regions_info:
            seq_info = SeqInformation()
            seq_info.from_dict(info.to_dict())
            try:
                if seq_info.strand == 'plus':
                    seq_info.start = seq_info.start - random.randint(upstream_range[0],upstream_range[1])
                    seq_info.end = seq_info.start + seq_length - 1
                else:
                    seq_info.end = seq_info.end + random.randint(upstream_range[0],upstream_range[1])
                    seq_info.start = seq_info.end - seq_length + 1
                if chroms_length[str(seq_info.chromosome_id)] > seq_info.end:
                    selected_regions.add(seq_info)
            except:
                pass
        return selected_regions