from . import seqs2dnn_data
from . import pyplot as plt
#visualize the exon probability along 5' end to 3' end DNA sequence
def visualization(model,x,true_y):
    for index in range(len(x)):
        target=x[index]
        data=model.predict(numpy.array([target]))[0]
        plt.plot(data)
        plt.plot(true_y[index])
        plt.title('index:'+str(index))
        plt.xlabel('nt')
        plt.ylabel('exon probability')
        plt.show()