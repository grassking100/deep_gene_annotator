"""This submodule provides library about visualize"""
import matplotlib.pyplot as plt
import numpy as np
import os,sys
import pandas as pd
from . import ModelHandler
from . import SeqConverter
from . import FastaConverter

def ann_visual(answer, annotation_types):
    """visualize the probability of each type along  sequence"""    
    answer_vec = []
    for type_ in annotation_types:
        answer_vec.append(answer[type_])
    x = list(range(np.array(answer_vec).shape[1]))
    plt.stackplot(x,answer_vec, labels=annotation_types)

def seq_predict_visual(model, seqs, annotation_types):
    """visualize the predicted probability of each type along sequence"""
    predict = np.transpose(model.predict(np.array([seqs])))
    data = dict(zip(annotations,predict))
    ann_visual(data,annotation_types)

def position_error(model, sequence, answer,annotation_types):
    """calculate the error of each type along sequence"""
    predict = np.transpose(model.predict(np.array([sequence]))[0])
    error = {}
    for index,type_ in enumerate(annotation_types):
        error[type_]=predict[index] - answer[type_]
    return  error

def error_visual(model, sequence, answer,annotation_types):
    """visualize the error of each type along sequence"""
    error_status = position_error(model, sequence, answer, annotation_types)
    for type_ in annotation_types:
        error = error_status[type_]
        plt.plot(error, label=type_)

def get_values_list(df,metric_type,ann_type=None):
    values_list = []
    metric = df[df['metric_type']==metric_type]
    if ann_type is not None:
        selected_metric = metric[metric['ann_type']==ann_type]
    else:
        selected_metric = metric[metric['metric_type']==metric_type]
    for type_ in ['train','validation']:
        selected=selected_metric[selected_metric['source']==type_]
        values_list+=selected.to_dict(orient='records')
    return values_list
def draw_metric_curve(data,annotations,line_type,title=None):
    index = 1
    def plot(title,ylabel,ylim=None):
        plt.title(title,fontsize=18)
        plt.xlabel('epoch',fontsize=18)
        plt.ylabel(ylabel,fontsize=18)
        plt.ylim(ylim)
        plt.xlim([0,200])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
    def new_plot(index):
        plt.subplot(len(annotations)+1,3,index)
        return index + 1
    fig = plt.figure(figsize=(20,20))
    legends = []
    for ann_type in annotations:
        for metric_type in ['precision','recall','f1']:
            metrics = {'train':[],'validation':[]}
            values_list = get_values_list(data,metric_type,ann_type)
            index=new_plot(index)
            for metric in values_list:
                plt.plot(list(metric['value']),line_type[metric['source']])
            plot(ann_type+"'s "+metric_type,metric_type,[0,1])
    index=new_plot(index)
    accs = get_values_list(data,'accuracy')
    for acc in accs:
        plt.plot(list(acc['value']),line_type[acc['source']])
    plot("accuracy",'accuracy',[0,1])
    index=new_plot(index)
    losses = get_values_list(data,'loss')
    for loss in losses:
        plt.plot(list(loss['value']),line_type[loss['source']])
    plot("loss",'loss')
    plt.subplots_adjust(top=.90,bottom =.1,wspace=.3,hspace=.8)
    plt.suptitle(title, fontsize=30)
    for type_ in annotations:
        fig.legend(['train','validation'], prop={'size': 20})
    fig.savefig('loss_accuracy_and_metrics_of_each_type_of_expand_8_9_10/'+title.replace(' ','_').replace(',','_')+".png")
def draw_metric_barplot(datas,annotations,title=None):
    index = 0
    fig,axes = plt.subplots(12,5)
    fig.set_size_inches(50, 50)
    def group_value(dfs, metric_type, ann_type=None):
        last_metrics = []
        for df in dfs:
            metrics = get_values_list(df,metric_type,ann_type)
            for metric in metrics:
                last_metric = {}
                for note in ['ann_type','metric_type','source','train_id','mode_id']:
                    last_metric[note] = metric[note]
                last_metric['value'] = list(metric['value'])[-1]
                last_metrics.append(last_metric)
        return last_metrics
    for train_id in range(8,11):
        for metric_type in ['f1','precision','recall']:
            for ann_type in annotations:
                dfs=datas[str(train_id)]
                last_metrics = group_value(dfs,metric_type,ann_type)
                df = pd.DataFrame(last_metrics).pivot(index='mode_id',columns='source',values='value')
                df.plot.bar(ax=axes[int(index/5),index%5],
                            title='expand_'+str(train_id)+' last '+ann_type+" "+metric_type+' of each mode',
                            legend=None,ylim=[0,1])
                index+=1 
        for metric_type in ['loss','accuracy']:
            ylim=[0,1]
            if metric_type=='loss':
                ylim=None
            dfs=datas[str(train_id)]
            last_metrics = group_value(dfs,metric_type)
            df = pd.DataFrame(last_metrics).pivot(index='mode_id',columns='source',values='value')
            df.plot.bar(ax=axes[int(index/5),index%5],
                        title='expand_'+str(train_id)+' last '+metric_type+' of each mode',
                        legend=None,ylim=ylim)
            index+=1
        index+=3
    fig.legend(['train','validation'])
    fig.subplots_adjust(top=.95,bottom =.1,wspace=.1,hspace=.5)
    fig.suptitle(title, fontsize=30)
    fig.savefig('loss_accuracy_and_metrics_of_each_type_of_expand_8_9_10/'+title.replace(' ','_').replace(',','_')+".png")
def metric_converter(pandas_data,sources,
                     annotation_types=None,
                     train_id=None,mode_id=None):
    def name(prefix,ann,metric):
        return prefix+ann+"_"+metric+"_layer"
    metrics = []
    prefix = ""
    for source in sources:
        if source=='validation':
            prefix='val_' 
        if annotation_types is not None:
            for type_ in annotation_types:
                tp = pandas_data[name(prefix,type_,"TP")]
                tn = pandas_data[name(prefix,type_,"TN")]
                fp = pandas_data[name(prefix,type_,"FP")]
                fn = pandas_data[name(prefix,type_,"FN")]
                recall =  tp/(tp+fn)
                precision =  tp/(tp+fp)
                f1 = 2*(precision*recall)/(precision+recall)
                metrics.append({'value':recall,'ann_type':type_,
                                'metric_type':'recall',
                                'source':source,'train_id':train_id,
                                'mode_id':mode_id})
                metrics.append({'value':precision,'ann_type':type_,
                                'source':source,
                                'metric_type':'precision',
                                'train_id':train_id,
                                'mode_id':mode_id})
                metrics.append({'value':f1,'ann_type':type_,
                                'source':source,
                                'metric_type':'f1',
                                'train_id':train_id,
                                'mode_id':mode_id})
        metrics.append({'value':pandas_data[prefix+'loss'],'source':source,
                        'ann_type':'global','metric_type':'loss',
                        'train_id':train_id,'mode_id':mode_id})   
        metrics.append({'value':pandas_data[prefix+'accuracy'],'source':source,
                        'ann_type':'global','metric_type':'accuracy',
                        'train_id':train_id,'mode_id':mode_id})
    return pd.DataFrame(metrics)