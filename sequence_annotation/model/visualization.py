"""This submodule provides library about visualize"""
import matplotlib.pyplot as plt
import numpy
def visualization(model, sequences, annotations):
    """visualize the exon probability along 5' end to 3' end DNA sequence"""
    if len(sequences) != len(annotations):
        raise "Length is not consistent"
    length_of_data = len(sequences)
    for index in range(length_of_data):
        target = sequences[index]
        data = model.predict(numpy.array([target]))[0]
        plt.plot(data)
        plt.plot(annotations[index])
        plt.title('index:'+str(index))
        plt.xlabel('nt')
        plt.ylabel('exon probability')
        plt.show()
