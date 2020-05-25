import numpy as np
from matplotlib import pyplot as plt

def plot_boxplot(lhs_values,rhs_values,lhs_name,rhs_name,metric_name,ylabel):
    _=plt.boxplot(x=[lhs_values,rhs_values],labels=[lhs_name,rhs_name])
    plt.xlabel('Method', fontsize=8)
    plt.ylabel(ylabel, fontsize=8)
    plt.title('{} boxplot'.format(metric_name), fontsize=8)
    
def plot_boxplots(lhs_result,rhs_result,lhs_name,rhs_name,metrics,ylabel):
    plt.style.use('ggplot')
    nrow=int(np.ceil(len(metrics)/2))
    figure, axes = plt.subplots(nrows=nrow, ncols=2)
    figure.tight_layout(pad=3.0)
    lhs_list = lhs_result.to_dict('list')
    rhs_list = rhs_result.to_dict('list')
    for index in range(0,len(metrics)):
        plt.subplot(nrow,2,index+1)
        metric_name = metrics[index]
        #metric_name = metric_names[index]
        lhs_values = lhs_list[metric_name]
        rhs_values = rhs_list[metric_name]
        plot_boxplot(lhs_values,rhs_values,lhs_name,rhs_name,metric_name,ylabel)
