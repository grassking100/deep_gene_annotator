"""This submodule provides library about visualize"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import SubplotHelper
def visual_ann_seq(seq):
    """visualize the count of each type along sequence"""    
    answer_vec = []
    for type_ in seq.ANN_TYPES:
        answer_vec.append(np.array([0.0]*seq.length))
    for index,type_ in enumerate(seq.ANN_TYPES):
        if seq.strand=='plus':
            answer_vec[index] += seq.get_ann(type_)
        else:
            answer_vec[index] += np.flip(seq.get_ann(type_),0)
    x = list(range(seq.length))
    plt.stackplot(x,answer_vec,labels=seq.ANN_TYPES)
    plt.legend(loc='upper right')

def visual_ann_seqs(seqs):
    """visualize the count of each type along sequences"""    
    answer_vec = []
    max_len = 0
    for seq in seqs:
        max_len = max(seq.length,max_len)
    for type_ in seqs.ANN_TYPES:
        answer_vec.append(np.array([0.0]*max_len))
    for seq in seqs:
        for index,type_ in enumerate(seqs.ANN_TYPES):
            if seq.strand=='plus':
                answer_vec[index][0:seq.length] += seq.get_ann(type_)
            else:
                answer_vec[index][0:seq.length] += np.flip(seq.get_ann(type_),0)
    x = list(range(np.array(answer_vec).shape[1]))
    plt.stackplot(x,answer_vec, labels=seqs.ANN_TYPES)
    plt.legend(loc='upper right')    

def seq_predict_visual(model, seqs, annotation_types):
    """visualize the predicted probability of each type along sequence"""
    predict = np.transpose(model.predict(np.array([seqs])))
    data = dict(zip(annotation_types,predict))
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

def metric_mean_converter(df):
    "Get mean of each metric in dataframe form"
    mean_list = []
    for metric_type in set(df['metric_type']): 
        for ann_type in set(df['ann_type']):
            if (ann_type=='global')!= (metric_type in ['loss','accuracy']):
                continue
            for id_ in set(df['id']):
                for source in set(df['source']):
                    last_metric = get_sub_dataframe(df,metric_type=metric_type,source=source,
                                                    ann_type=ann_type,index=-1,id_=id_)
                    mean_value = last_metric['value'].mean()
                    series = {'source':source,'ann_type':ann_type,'id':id_,
                              'metric_type':metric_type,'value':mean_value}
                    mean_list.append(pd.Series(series))
    return pd.concat(mean_list,axis=1).T

def get_values_list(df,metric_type,sources,ann_type=None,mode_id=None):
    values_list = []
    metric = df[df['metric_type']==metric_type]
    if ann_type is not None:
        metric = metric[metric['ann_type']==ann_type]
    if mode_id is not None:
        metric = metric[metric['mode_id']==mode_id]
    for type_ in sources:
        selected=metric[metric['source']==type_]
        values_list+=selected.to_dict(orient='records')
    return values_list

def draw_metric_curve(data,annotations,sources,line_types,scale_y_axis=True):
    helper = SubplotHelper(6,3,50,50)
    helper.set_axes_setting(xlabel_params={'xlabel':'epoch','fontsize':30},
                            ylabel_params={'ylabel':'metric','fontsize':30})
    for ann_type in annotations:
        for metric_type in ['precision','recall','f1']:
            metrics = {'train':[],'validation':[]}
            values_list = get_values_list(data,metric_type,ann_type=ann_type,sources=sources)
            ax = helper.get_ax()
            ax.set_title(ann_type+"'s "+metric_type,size=30)
            for metric in values_list:
                ax.plot(list(metric['value']),line_types[metric['source']])
                #ax.set_xlim([0,200])
                if scale_y_axis:
                    ax.set_ylim([0,1])
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(30)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(30)
    accs = get_values_list(data,'accuracy',sources=sources)
    ax = helper.get_ax()
    ax.set_title("accuracy",size=30)
    for acc in accs:
        ax.plot(list(acc['value']),line_types[acc['source']])
        #ax.set_xlim([0,200])
        if scale_y_axis:
            ax.set_ylim([0,1])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(30)
    losses = get_values_list(data,'loss',sources=sources)
    ax = helper.get_ax()
    ax.set_title("loss",size=30)
    for loss in losses:
        ax.plot(list(loss['value']),line_types[loss['source']])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(30)
    for type_ in annotations:
        helper.fig.legend(sources, prop={'size': 40})
    return helper.fig,helper.axes

def draw_each_class_barplot(df,soruces):
    ids = set(df['id'])
    metric_types = set(df['metric_type'])
    helper = SubplotHelper(len(metric_types),len(ids),50,50)
    for metric_type in metric_types:
        for id_ in sorted(ids):
            ylim=[0,1]
            if metric_type=='loss':
                ylim=None
            selected_df = get_sub_dataframe(df,metric_type=metric_type,id_=id_) 
            pivot_df = selected_df.pivot('ann_type','source','value')[soruces]
            ax = helper.get_ax()
            pivot_df.plot.bar(ax=ax,rot=0,ylim=ylim)
            ax.set_title("expand_"+str(id_)+" "+metric_type,size=30)
            helper.fig.subplots_adjust(top=.95,bottom =.1,wspace=.2,hspace=.6)
            ax.set_xlabel('annotation type', fontsize=30)
            ax.set_ylabel('metric', fontsize=30)
            ax.tick_params(axis='both',labelsize=30)
            ax.tick_params(axis='x',rotation=10)
            ax.legend(fontsize=30,mode="expand",bbox_to_anchor=(0.,-.5, 1., -.5),
                      borderaxespad=0.,ncol=2, loc='lower center')   

def draw_each_setting_barplot(df,soruces):
    ann_types = set(df['ann_type'])
    metric_types = ['precision','recall','f1','loss','accuracy']
    helper = SubplotHelper(len(metric_types)-1,len(ann_types)-1,50,50)
    for metric_type in metric_types:
        for ann_type in ann_types:
            if (ann_type=='global') == (metric_type in ['loss','accuracy']):
                ylim=[0,1]
                if metric_type=='loss':
                    ylim=None
                selected_df = get_sub_dataframe(df,metric_type=metric_type,ann_type=ann_type) 
                pivot_df = selected_df.pivot('id','source','value')[soruces]
                ax = helper.get_ax()
                pivot_df.plot.bar(ax=ax,rot=0,ylim=ylim)
                ax.set_title(str(ann_type)+" "+metric_type,size=30)
                helper.fig.subplots_adjust(top=.95,bottom =.1,wspace=.2,hspace=.4)
                ax.set_xlabel('id', fontsize=30)
                ax.set_ylabel('metric', fontsize=30)
                ax.tick_params(axis='both',labelsize=30)
                ax.legend(fontsize=30,mode="expand",bbox_to_anchor=(0.,-.3, 1., -.3),
                          borderaxespad=0.,ncol=2, loc='lower center')

def draw_loss_curve(df,line_type,colors):
    modes = set(df['mode_id'])
    helper = SubplotHelper(len(modes),2,20,10*len(modes))
    helper.set_axes_setting(xlabel_params={'xlabel':'epoch','fontsize':30},
                            ylabel_params={'ylabel':'metric','fontsize':30})
    color_setting = {}
    id_list = list(set(df['id']))
    id_list.sort()
    for id_,color in zip(id_list,colors):
        color_setting[id_]=color
    for mode in modes:
        for source in ['train','validation']:
            values_list = get_values_list(df,'loss',sources=[source], mode_id=mode)
            ax = helper.get_ax()
            for item in values_list:
                id_ = "expand_"+str(item['id'])
                ax.set_title("Mode "+str(mode)+" "+source+" loss per epoch in each setting")
                ax.semilogy(list(item['value']),label = id_, color=color_setting[item['id']])
                ax.ylabel="loss"
                ax.xlabel='epoch'
                ax.legend(loc='best')  
    return helper.fig,helper.axes

def metric_converter(df,sources,prefixes,annotation_types=None,id_=None,mode_id=None):
    """Create dataframe of annotation metric and global metric depends on input"""
    def name(prefix,ann,metric):
        return prefix+ann+"_"+metric+"_layer"
    metrics = []
    for prefix,source in zip(prefixes,sources):
        if annotation_types is not None:
            for type_ in annotation_types:
                metric_value = {}
                tp = df[name(prefix,type_,"TP")]
                tn = df[name(prefix,type_,"TN")]
                fp = df[name(prefix,type_,"FP")]
                fn = df[name(prefix,type_,"FN")]
                recall =  tp/(tp+fn)
                precision =  tp/(tp+fp)
                f1 = 2*(precision*recall)/(precision+recall)
                metric_value['recall']=recall
                metric_value['precision']=precision
                metric_value['f1']=f1
                for metric_type,value in metric_value.items():
                    metrics.append({'value':value,'ann_type':type_,'metric_type':metric_type,
                                    'source':source,'id':id_,'mode_id':mode_id})
        metric_value = {}            
        metric_value['loss']=df[prefix+'loss']
        metric_value['accuracy']=df[prefix+'accuracy']
        for metric_type,value in metric_value.items():
            metrics.append({'value':value,'ann_type':'global','metric_type':metric_type,
                            'source':source,'id':id_,'mode_id':mode_id})
    return pd.DataFrame(metrics)

def get_sub_dataframe(df,metric_type=None,source=None,ann_type=None,mode_id=None,index=None,id_=None):
    """Get dataframe of data which are selected in dataframe"""
    if metric_type is not None:
        df = df[df['metric_type']==metric_type]
    if ann_type is not None:
        df = df[df['ann_type']==ann_type]
    if mode_id is not None:
        df = df[df['mode_id']==mode_id]
    if id_ is not None:
        df = df[df['id']==id_]
    if index is not None:
        temp = []
        for row in list(df['value']):
            if isinstance(row,float):
                temp.append(row)
            else:
                temp.append(list(row)[index])
        df['value']=temp
    if source is not None:
        df=df[df['source']==source]
    return df