import configparser
import argparse
#load input parameter
parser = argparse.ArgumentParser(description="Program will test the mode by the data which is input")
parser.add_argument('-f','--fasta',help='File\'s path to test', required=True)
parser.add_argument('-m','--model',help='Model\'s path to load', required=True)
parser.add_argument('-o','--output',help='Output\'s path to save', required=True)
parser.add_argument('-n','--notes',help='Notes to add in output file', required=False)
args = parser.parse_args()
import keras
from keras.models import load_model
import os, errno
from Exon_intron_finder.Exon_intron_finder import tensor_end_with_terminal_binary_accuracy,tensor_end_with_terminal_binary_crossentropy
from Fasta_handler.Fasta_handler import seqs_index_selector,fastas2arr,seqs2dnn_data
import Exon_intron_finder
import numpy
import csv
#load data
input_file=args.fasta
print("Load data from file:"+input_file)
(names,seqs)=fastas2arr(input_file)
x,y=seqs2dnn_data(seqs)
x_testing=keras.preprocessing.sequence.pad_sequences(x, maxlen=None,padding='post')
y_testing=keras.preprocessing.sequence.pad_sequences(y, maxlen=None,padding='post',value=-1)
#load model
file_root_name=args.model
print("Load model from:"+file_root_name)
model=load_model(file_root_name+".h5",custom_objects={'tensor_end_with_terminal_binary_crossentropy':Exon_intron_finder.Exon_intron_finder.tensor_end_with_terminal_binary_crossentropy,'tensor_end_with_terminal_binary_accuracy':Exon_intron_finder.Exon_intron_finder.tensor_end_with_terminal_binary_accuracy})
#testing model
print("Start testing")
results=[]
index=1
length=len(x)
for i in range(len(x_testing)):
    print("Progress:"+str(round(100*index/length,2))+"%")
    index+=1
    results.append(model.evaluate(numpy.array([x_testing[i]]),numpy.array([y_testing[i]]),batch_size=1,verbose=0))
#save result(id,loss,accuracy)
output_file=args.output
print("Save file to "+output_file)
notes=args.notes
with open(output_file,"w") as output_handler:
    writer=csv.writer(output_handler)
    if notes is not None:
        writer.writerow("#"+notes)
    writer.writerow(["id","loss","accuracy"])
    index=0
    for result in results:
        id=names[index]
        loss=result[0]
        accuracy=result[1]
        writer.writerow([id,loss,accuracy])
        index+=1