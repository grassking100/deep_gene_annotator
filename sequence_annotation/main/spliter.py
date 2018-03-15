dataset_indice = [
    [2,11,8,21],
    [5,17,15,6],
    [1,18,19,20],
    [13,12,4],
    [3,9,14],
    [10,7,16]
]
cv_number = len(dataset_indice) - 1
mode_indice = []
for dataset_id,dataset_index in enumerate(dataset_indice):
    dataset = []
    for index in dataset_index:
        read_path = 'tetraodon_8_0/2018_03_07/chrom_{index}_seq.gtf'.format(index=index)
        dataset.append(pd.read_csv(read_path,header=None,sep='\t'))
    saved_file=pd.concat(dataset)
    write_path='tetraodon_8_0/2018_03_07/dataset_{index}.gtf'.format(index=dataset_id)
    saved_file.to_csv(write_path,sep='\t',header=None,index=False)
    #parse file
    #save file
for index in range(1,cv_number+1):
    mode_index = list(range(1,cv_number+1))
    mode_index.remove(index)
    fold = []
    for i in mode_index:
        read_path = 'tetraodon_8_0/2018_03_07/dataset_{index}.gtf'.format(index=i)
        fold.append(pd.read_csv(read_path,header=None,sep='\t'))
    saved_file=pd.concat(fold)
    write_path='tetraodon_8_0/2018_03_07/fold_{index}.gtf'.format(index=index)
    saved_file.to_csv(write_path,sep='\t',header=None,index=False)
    #parse file
    #save file