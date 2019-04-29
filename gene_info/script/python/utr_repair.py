import pandas as pd
def get_df_with_id(df_dict):
    data = []
    for item in df_dict:
        ids = item[8].split(';')
        type_ = item[2]
        item_id = ""
        item_name = ""
        item_parent = ""
        for id_ in ids:
            if id_.startswith("Parent"):
                item_parent = id_[7:]
            if id_.startswith("ID"):
                item_id = id_[3:]
            if id_.startswith("Name"):
                item_name = id_[5:]
        item_parent = item_parent.split(",")
        item_id = item_id.split(",")
        for parent in item_parent:
            for id_ in item_id:
                copied = dict(item)
                copied['id'] = id_
                copied['parent'] = parent
                copied['name'] = item_name
                data.append(copied)
    df = pd.DataFrame.from_dict(data)
    return df
class GFFRepairer:
    def __init__(self):
        self.exon_subtype = ['five_prime_UTR','three_prime_UTR','CDS']
    def repair(self,df):
        df_dict = df.to_dict('rercord')
        df = get_df_with_id(df_dict)
        gene_ids = df[df[2]=='mRNA']
        id_table = {}
        for item in gene_ids.to_dict('recrod'):
            id_table[item['id']] = item['parent']
        CDS = df[df[2]=='CDS']
        coding_mRNA_id = set(CDS['parent'])
        coding_gene_id = set([id_table[id_] for id_ in coding_mRNA_id])
        coding_exon_subtype_df = df[(df[2].isin(self.exon_subtype)) & (df['parent'].isin(coding_mRNA_id))]
        coding_mRNA_df = df[(df[2]=='mRNA') & (df['id'].isin(coding_mRNA_id))]
        coding_gene_df = df[(df[2]=='gene') & (df['id'].isin(coding_gene_id))]
        coding_protein_df = df[(df[2]=='protein') & (df['name'].isin(coding_mRNA_id))]
        exons_info = []
        groups = coding_exon_subtype_df.groupby('parent')
        for name in coding_mRNA_id:
            group = groups.get_group(name)
            exons_info += self.extract_exon(group)
        exons = pd.DataFrame.from_dict(exons_info)
        return pd.concat([coding_gene_df,coding_mRNA_df,exons,coding_exon_subtype_df,coding_protein_df])
    def extract_exon(self,group):
        info_ = dict(group.iloc[0,:].to_dict())
        info_[2] = 'exon'
        list_group = group.to_dict('list')
        starts = list_group[3]
        ends = list_group[4]
        sizes = [end-start+1 for start,end in zip(starts,ends)]
        sum_size = sum(sizes)
        exons_info = []
        indice = np.argsort(starts)
        exon_start = None
        exon_end = None
        update = False
        for index in indice:
            start = starts[index]
            end = ends[index]
            if exon_start is None:
                exon_start = start
            else:
                continue_ = (start == exon_end+1)
                if not continue_:
                    info = dict(info_)
                    info[3] = exon_start
                    info[4] = exon_end
                    exon_start = start
                    exons_info.append(info)
            exon_end = end
        info = dict(info_)
        info[3] = exon_start
        info[4] = exon_end
        exons_info.append(info)
        if sum_size != sum([item[4]-item[3]+1 for item in exons_info]):
            raise Exception()
        return exons_info
    def gff_info2bed_info(mRNA,exon,orf_info):
        mRNA_start = mRNA[3]
        exon_starts = exon[3]
        exon_ends = exon[4]
        indice = np.argsort(exon_starts)
        exon_starts = [exon_starts[index] for index in indice]
        exon_ends = [exon_ends[index] for index in indice]
        exon_size = [str(end-start+1) for start,end in zip(exon_starts,exon_ends)]
        exon_rel_starts = [str(start-mRNA_start) for start in exon_starts]
        info = {}
        info['id'] = mRNA['id']
        info['start'] = mRNA_start - 1
        info['end'] = mRNA[4]
        info['strand'] = mRNA[6]
        info['color'] = '.'
        info['chr'] = mRNA[0][3:]
        info['block_size'] = len(exon_size)
        info['exon_size'] = ",".join(exon_size)
        info['exon_rel_starts'] = ",".join(exon_rel_starts)
        info['score'] = '.'
        info['orf_start'] = orf_info['orf_start'] - 1
        info['orf_end'] = orf_info['orf_end']
        return info
    def extract_orf(CDS_group):
        info_ = dict(CDS_group.iloc[0,:].to_dict())
        id_ = info_['id']
        orf_start = min(CDS_group[3])
        orf_end = max(CDS_group[4])
        return {'id':id_,'orf_start':orf_start,'orf_end':orf_end}

if __name__ =='__main__':
    df_ = pd.read_csv('Araport11_GFF3_genes_transposons.201606.gff',sep='\t',header=None,comment ='#')
    repairer = GFFRepairer()
    df = repairer.repair(df_)
    df.iloc[:,:9].to_csv('Araport11_GFF3_genes_transposons.201606_coding_repair_2019_04_06.gff',header=None,index=None,sep='\t')
    gene = df[df[2]=='mRNA']
    exon = df[df[2]=='exon']
    gene_bed = gene[[0,3,4,'id',5,6]]
    gene_bed[0] = gene_bed[0].str.replace('Chr','')
    gene_bed[3] -= 1
    exon_bed = exon[[0,3,4,'parent',5,6]]
    exon_bed[0] = exon_bed[0].str.replace('Chr','')
    exon_bed[3] -= 1
    gene_bed.to_csv('Araport11_coding_gene_2019_04_07.bed',header=None,index=None,sep='\t')
    exon_bed.to_csv('Araport11_coding_exon_2019_04_07.bed',header=None,index=None,sep='\t')
    mRNAs = df[df[2]=='mRNA']
    ids = set(mRNAs['id'])
    mRNAs = mRNAs.groupby('id')
    CDS = df[df[2].isin(['CDS'])].groupby('parent')
    exons = df[df[2].isin(['exon'])].groupby('parent')
    bed = []
    for id_ in ids:
        group = CDS.get_group(id_)
        orf_info = extract_orf(group)
        mRNA = mRNAs.get_group(id_).to_dict('record')[0]
        exon = exons.get_group(id_).to_dict('list')
        bed_info = gff_info2bed_info(mRNA,exon,orf_info)
        bed.append(bed_info)
    bed = pd.DataFrame.from_dict(bed)
    bed = bed[['chr','start','end','id','score','strand','orf_start','orf_end','color','block_size','exon_size','exon_rel_starts']]
    bed['chr'] = 'Chr'+bed['chr'].astype(str)
    bed.to_csv('Araport11_GFF3_genes_transposons.201606_coding_repair_2019_04_07.bed',header=None,index=None,sep='\t')