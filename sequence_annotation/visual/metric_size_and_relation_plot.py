import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from sklearn import linear_model
from sklearn.metrics import r2_score
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_json

def get_metric_size_relation_type(optuna_trained_root):
    trial_list_path = os.path.join(optuna_trained_root, 'trial_list.tsv')
    trial_list = list(pd.read_csv(trial_list_path, sep='\t')['path'])
    data = {}
    for trial_name in trial_list:
        trial_root = os.path.join(optuna_trained_root, trial_name)
        result_path = os.path.join(trial_root,
                                   '{}_result.json'.format(trial_name))
        if os.path.exists(result_path):
            param_data = os.path.join(trial_root, 'settings',
                                      'model_param_num.txt')
            with open(param_data, 'r') as fp:
                param_size = int(fp.read().split(':')[1])

            result = read_json(result_path)
            data[trial_name] = {
                'param_size': param_size,
                'value': result['value'],
                'relation_type': result['params']['relation_type']
            }
    return data


def plot_metric_size_relation_type(result,
                                   metric_name=None,
                                   show_percentage=False):
    metric_name = metric_name or 'metric'
    cluster = {}
    sizes = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for item in result.values():
        relation_type = item['relation_type']
        if relation_type not in cluster:
            cluster[relation_type] = {}
            cluster[relation_type]['param_size'] = []
            cluster[relation_type]['value'] = []
        size = np.log10(item['param_size'])
        value = item['value']
        sizes.append(size)
        if show_percentage:
            value *= 100
        cluster[relation_type]['param_size'].append(size)
        cluster[relation_type]['value'].append(value)

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
        plt.plot(sorted_size,
                 y_pred_for_line,
                 '-',
                 color=label_color[rel_type],
                 label=label)
    plt.title(
        "Relation between {} and model's\n required-gradient parameter number".
        format(metric_name))
    if show_percentage:
        plt.ylabel("y={}(%)".format(metric_name))
    else:
        plt.ylabel("y={}".format(metric_name))

    plt.xlabel("x=log10(size)")
    plt.legend()


def main(optuna_trained_root, saved_path):
    result = get_metric_size_relation_type(optuna_trained_root)
    plot_metric_size_relation_type(result, 'macro F1', True)
    plt.savefig(saved_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t", "--optuna_trained_root", required=True)
    parser.add_argument("-s", "--saved_path", required=True)

    args = parser.parse_args()
    setting = vars(args)

    main(**setting)
