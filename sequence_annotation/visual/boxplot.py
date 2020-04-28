import numpy as np
from matplotlib import pyplot as plt

def plot_boxplot(lhs_values,rhs_values,lhs_name,rhs_name,metric_name):
    _=plt.boxplot(x=[lhs_values,rhs_values],labels=[lhs_name,rhs_name])
    plt.xlabel('Method')
    plt.ylabel(metric_name)
    plt.title('{} boxplot'.format(metric_name))
    
def plot_boxplots(lhs_result,rhs_result,lhs_name,rhs_name,metrics,metric_names):
    plt.style.use('ggplot')
    nrow=int(np.ceil(len(metrics)/2))
    figure, axes = plt.subplots(nrows=nrow, ncols=2)
    figure.tight_layout(pad=5.0)
    lhs_list = lhs_result.to_dict('list')
    rhs_list = rhs_result.to_dict('list')
    for index in range(0,len(metrics)):
        plt.subplot(nrow,2,index+1)
        metric = metrics[index]
        metric_name = metric_names[index]
        lhs_values = lhs_list[metric]
        rhs_values = rhs_list[metric]
        plot_boxplot(lhs_values,rhs_values,lhs_name,rhs_name,metric_name)
