from . import SeqIO
#read fasta file and return array of sequnece and name
#if number is negative,then all the sequneces will be read
#otherwirse read part of sequneces,the number indicate how many to read
def fasta2seqs(file_name,number=-1):
    fasta_sequences = SeqIO.parse(open(file_name),'fasta')
    names=[]
    seqs=[]
    counter=0
    for fasta in fasta_sequences:
        if (number<=0)|(counter<number): 
            name,seq=fasta.id,(str)(fasta.seq)
            names.append(name)
            seqs.append(seq)
            counter+=1
        else:
            break
    return(names,seqs)
#read fasta file and return the data format which tensorflow can input
def fasta2dnn_data(file_name,number=-1,safe=False):
    (names,seqs)=fastas2arr(file_name,number)
    return seqs2dnn_data(seqs,safe)

#def fasta_extractor(fasta_file,cross_validation_index,testing_index,shuffle=True):
#    records=[]
#    names,seqs =fasta2arr(fasta_file)
#    for name,seq in zip(names,seqs):
#        temp = SeqRecord(Seq(seq,IUPACAmbiguousDNA), id = name)
#        records.append(temp)
#    with open(fasta_file, "w") as output_handle:
#        for i in len(records):
#            SeqIO.write(records[i], output_handle, "fasta")
