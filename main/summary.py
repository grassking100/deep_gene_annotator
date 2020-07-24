def get_all_results(root,data_type=None):
    result = {}
    types = ['base','block','distance','site']
    file_names = ['base_performance.json','block_performance.json','abs_diff.json','site_matched.json']
    columns = [
        'base_F1_exon','base_F1_intron','base_F1_other','base_macro_F1',
        'block_exon_F1','block_gene_F1','block_intron_F1','block_intron_chain_F1',
        'distance_TSS','distance_cleavage_site','distance_splicing_acceptor_site','distance_splicing_donor_site',
        'site_TSS','site_cleavage_site','site_splicing_acceptor_site','site_splicing_donor_site'
    ]
    for type_,file_name in zip(types,file_names):
        for name in sorted(os.listdir(root)):
            if data_type is not None:
                path = os.path.join(root,name,data_type,file_name)
            else:
                path = os.path.join(root,name,file_name)
            #print(path)
            if os.path.exists(path):
                with open(path,'r') as fp:
                    data_ = json.load(fp)
                data = {}
                if type_ == 'site':
                    for key,value in data_['F1'].items():
                        data[type_+"_"+key] = value
                elif type_ == 'distance':
                    for key,value in data_['mean'].items():
                        data[type_+"_"+key] = value    
                else:
                    for key,value in data_.items():
                        data[type_+"_"+key] = value

                if name not in result:
                    result[name] = {}
                result[name].update(data)
    result = pd.DataFrame.from_dict(result).T[columns].T
    return result