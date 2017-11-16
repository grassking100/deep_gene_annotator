from . import SeqIO
#read fasta file and return array of sequnece and name
#if number is negative,then all the sequneces will be read
#otherwirse read part of sequneces,the number indicate how many to read
def fasta2arr(file_name,number=-1):
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
