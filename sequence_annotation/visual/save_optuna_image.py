import os
import sys
import optuna
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder,read_json
from sequence_annotation.visual.visual import partial_dependence_plot, plot_optim_history, boxplot


def parsed_result(optuna_trained_root):
    space = read_json(os.path.join(optuna_trained_root, 'space_config.json'))
    hyperparam_names = []
    for type_,values in space.items():
        if values['value'] is None:
            hyperparam_names.append(type_)
    trial_list_path = os.path.join(optuna_trained_root, 'trial_list.tsv')
    trial_names = list(pd.read_csv(trial_list_path, sep='\t')['path'])
    data = {}
    for trial_name in trial_names:
        trial_root = os.path.join(optuna_trained_root, trial_name)
        result_path = os.path.join(trial_root,'{}_result.json'.format(trial_name))
        if os.path.exists(result_path):
            param_data = os.path.join(trial_root, 'settings',
                                      'model_param_num.txt')
            with open(param_data, 'r') as fp:
                param_size = int(fp.read().split(':')[1])

            result = read_json(result_path)
            data[trial_name] = {
                'param_size': param_size,
                'value': result['value'],
                'params': result['params']
            }
    return data,hyperparam_names


def plot_metric_size_with_type(result,hyperparam_name,metric_name,
                               show_percentage=False):
    cluster = {}
    sizes = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for item in result.values():
        type_ = item['params'][hyperparam_name]
        if type_ not in cluster:
            cluster[type_] = {}
            cluster[type_]['param_size'] = []
            cluster[type_]['value'] = []
        size = np.log10(item['param_size'])
        value = item['value']
        sizes.append(size)
        if show_percentage:
            value *= 100
        cluster[type_]['param_size'].append(size)
        cluster[type_]['value'].append(value)

    sorted_size = np.array(sorted(sizes))
    sorted_types = sorted(cluster.keys())
    label_color = {}
    for index, type_ in enumerate(sorted_types):
        label_color[type_] = colors[index % len(sorted_types)]

    for rel_type in sorted_types:
        item = cluster[rel_type]
        x = np.array(item['param_size'])
        y = item['value']
        model = linear_model.LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        r2 = r2_score(y, y_pred)
        y_pred_for_line = model.predict(sorted_size.reshape(-1, 1))
        label = "{} ,y={:.3f}*x+{:.3f}, R={:.2f}".format(
            rel_type, model.coef_[0], model.intercept_, np.sqrt(r2))
        plt.plot(x, y, '.', color=label_color[rel_type])
        plt.plot(sorted_size,y_pred_for_line,'-',
                 color=label_color[rel_type],label=label)
    plt.title(
        "The relation plot about {}, model's\n "
        "required-gradient parameter number, and hyperparameter {}".
        format(metric_name,hyperparam_name))
    if show_percentage:
        plt.ylabel("y={}(%)".format(metric_name))
    else:
        plt.ylabel("y={}".format(metric_name))

    plt.xlabel("x=log10(required-gradient parameter number)")
    plt.legend()


def save_optuna_image(study, output_root):
    create_folder(output_root)
    df = study.trials_dataframe()
    plt.clf()
    boxplot(df,
            'params_relation_type',
            ylabel='macro F1',
            title='The boxplot '
            'of relation block types and losses')
    plt.savefig(os.path.join(output_root, 'relation_type_pdp.png'), bbox_inches = 'tight',pad_inches = 0)

    plt.clf()
    partial_dependence_plot(df,
                            'params_cnn_num',
                            ylabel='macro F1',
                            title='The partial '
                            'dependence plot of cnn layer number',
                            xlabel='cnn layer number')
    plt.savefig(os.path.join(output_root, 'cnn_num_pdp.png'), bbox_inches = 'tight',pad_inches = 0)

    plt.clf()
    partial_dependence_plot(
        df,
        'params_cnn_out',
        ylabel='macro F1',
        title='The partial '
        'dependence plot of cnn output channel number of each layer',
        xlabel='cnn output channel number of each layer')
    plt.savefig(os.path.join(output_root, 'cnn_out_pdp.png'), bbox_inches = 'tight',pad_inches = 0)

    plt.clf()
    partial_dependence_plot(df,
                            'params_kernel_size',
                            ylabel='macro F1',
                            title='The partial '
                            'dependence plot of cnn kernel size',
                            xlabel='cnn kernel size')
    plt.savefig(os.path.join(output_root, 'kernel_size_pdp.png'), bbox_inches = 'tight',pad_inches = 0)

    plt.clf()
    partial_dependence_plot(
        df,
        'params_rnn_hidden',
        ylabel='macro F1',
        title='The partial '
        'dependence plot of size of each RNN layer (per direction)',
        xlabel='hidden size of each RNN layer (per direction)')
    plt.savefig(os.path.join(output_root, 'rnn_hidden_pdp.png'), bbox_inches = 'tight',pad_inches = 0)

    plt.clf()
    partial_dependence_plot(df,
                            'params_rnn_num',
                            ylabel='macro F1',
                            title='The partial dependence '
                            'plot of number of RNN layer',
                            xlabel='number of RNN layer')
    plt.savefig(os.path.join(output_root, 'rnn_num_pdp.png'), bbox_inches = 'tight',pad_inches = 0)

    plt.clf()
    plot_optim_history(study)
    plt.savefig(os.path.join(output_root, 'optim_history.png'), bbox_inches = 'tight',pad_inches = 0)

    with open(os.path.join(output_root, 'best_trial.txt'), 'w') as fp:
        fp.write(str(study.best_trial))

    with open(os.path.join(output_root, 'trial_number.txt'), 'w') as fp:
        fp.write(str(len(df)))


def main(optuna_trained_root, output_root):
    study = optuna.load_study(
        "seq_ann", 'sqlite:///{}/trials.db'.format(optuna_trained_root))
    save_optuna_image(study, output_root)
    result,hyperparam_names = parsed_result(optuna_trained_root)
    for name in hyperparam_names:
        plt.cla()
        plot_metric_size_with_type(result,name, 'macro F1',True)
        output_path = os.path.join(output_root,"size_type_relation_plot_{}.png".format(name))
        plt.savefig(output_path, bbox_inches = 'tight',pad_inches = 0)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--optuna_trained_root", required=True)
    parser.add_argument("-o", "--output_root", required=True)

    args = parser.parse_args()
    setting = vars(args)

    main(**setting)
