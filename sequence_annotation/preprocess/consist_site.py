import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import get_gff_with_attribute, get_gff_with_updated_attribute
from sequence_annotation.utils.utils import write_gff, read_gff


def consist_(data, by, ref_value, drop_duplicated):
    returned = []
    ref_names = set(data[by])
    sectors = data.groupby(by)
    for name in ref_names:
        sector = sectors.get_group(name)
        max_value = max(sector[ref_value].astype(float))
        sector = sector.to_dict('record')
        list_ = []
        for item in sector:
            if float(item[ref_value]) == max_value:
                list_.append(item)
        true_count = len(list_)
        if drop_duplicated:
            if true_count == 1:
                returned += list_
        else:
            returned += list_
    return returned


def consist(data, by, ref_value, drop_duplicated):
    strands = set(data['strand'])
    chrs = set(data['chr'])
    return_data = []
    for strand_ in strands:
        for chr_ in chrs:
            subdata = data[(data['strand'] == strand_) & (data['chr'] == chr_)]
            consist_data = consist_(subdata, by, ref_value, drop_duplicated)
            return_data += consist_data
    df = pd.DataFrame.from_dict(return_data)
    return df


def _read(path):
    gff = get_gff_with_attribute(read_gff(path))
    coord_ref_id = gff['chr'].astype(str) + "_" + gff['strand'].astype(
        str) + "_" + gff['start'].astype(str) + "_" + gff['ref_name'].astype(
            str)
    coord_id = gff['chr'].astype(str) + "_" + gff['strand'].astype(
        str) + "_" + gff['start'].astype(str)
    gff = gff.assign(coord_ref_id=coord_ref_id)
    gff = gff.assign(coord_id=coord_id)
    gff = preserved_max_by_coord_ref_id(gff, 'ref_name', 'experimental_score')
    return gff


def get_ophan_on_ld_data(on_long_dist, on_external_UTR):
    return on_long_dist[~on_long_dist['coord_id'].
                        isin(on_external_UTR['coord_id'])]


def get_not_same_coord_ref_id_data(compared, comparing):
    return compared[~compared['coord_ref_id'].isin(comparing['coord_ref_id'])]


def preserved_max_by_coord_ref_id(gff, group_by, max_name):
    index_list = []
    for _, group in gff.groupby('coord_ref_id'):
        index = group[group[max_name] == max(group[max_name])].index[0]
        index_list.append(index)
    return gff.loc[index_list].copy()


def get_consist_site(on_external_UTR, on_long_dist, on_transcript, UTR_target,
                     remove_conflict):
    #Create orhpan data
    on_orphan = get_ophan_on_ld_data(on_long_dist, on_external_UTR)
    data = [on_external_UTR, on_orphan]
    if remove_conflict:
        #Add signal in transcript but not in external UTR to "other"
        other = get_not_same_coord_ref_id_data(on_transcript, on_external_UTR)
        data.append(other)

    data = pd.concat(data, sort=False).reset_index(drop=True)
    #Get consist site of every transcript
    data = consist(data,
                   'ref_name',
                   'experimental_score',
                   drop_duplicated=True)
    consist_site = data[data['feature'].isin([UTR_target])]
    return consist_site


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(
        description=
        "This program will output sites data which are valid in RNA external UTR"
        + ", and have no strogner signal in orphan site")
    parser.add_argument("--external_five_UTR_tss_path", required=True)
    parser.add_argument("--external_three_UTR_cs_path", required=True)
    parser.add_argument("--long_dist_tss_path", required=True)
    parser.add_argument("--long_dist_cs_path", required=True)
    parser.add_argument("--transcript_tss_path", required=True)
    parser.add_argument("--transcript_cs_path", required=True)
    parser.add_argument("-s", "--saved_root", required=True)
    parser.add_argument("--remove_transcript_external_UTR_conflict",
                        action='store_true')
    args = parser.parse_args()
    safe_tss_path = os.path.join(args.saved_root, 'safe_tss.gff3')
    safe_cs_path = os.path.join(args.saved_root, 'safe_cs.gff3')
    exists = [os.path.exists(path) for path in [safe_tss_path, safe_cs_path]]
    if all(exists):
        print("Result files are already exist, procedure will be skipped.")
    else:
        ###Read file###
        external_five_UTR_tss = _read(args.external_five_UTR_tss_path)
        external_three_UTR_cs = _read(args.external_three_UTR_cs_path)
        ld_tss = _read(args.long_dist_tss_path)
        ld_cs = _read(args.long_dist_cs_path)
        transcript_tss = _read(args.transcript_tss_path)
        transcript_cs = _read(args.transcript_cs_path)

        consist_tss_site = get_consist_site(
            external_five_UTR_tss, ld_tss, transcript_tss,
            "external_5_UTR_TSS", args.remove_transcript_external_UTR_conflict)
        consist_cs = get_consist_site(
            external_three_UTR_cs, ld_cs, transcript_cs, 'external_3_UTR_CS',
            args.remove_transcript_external_UTR_conflict)

        ###Write data###)
        write_gff(get_gff_with_updated_attribute(consist_tss_site),
                  safe_tss_path)
        write_gff(get_gff_with_updated_attribute(consist_cs), safe_cs_path)
