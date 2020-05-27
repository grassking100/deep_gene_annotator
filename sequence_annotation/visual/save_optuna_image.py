import os
import sys
import optuna
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder
from sequence_annotation.visual.visual import partial_dependence_plot, plot_optim_history, boxplot

def save_optuna_image(study, output_root):
    create_folder(output_root)
    df = study.trials_dataframe()
    plt.clf()
    boxplot(df,
            'params_relation_type',
            ylabel='macro F1',
            title='The boxplot '
            'of relation block types and losses')
    plt.savefig(os.path.join(output_root, 'relation_type_pdp.png'))

    plt.clf()
    partial_dependence_plot(df,
                            'params_cnn_num',
                            ylabel='macro F1',
                            title='The partial '
                            'dependence plot of cnn layer number',
                            xlabel='cnn layer number')
    plt.savefig(os.path.join(output_root, 'cnn_num_pdp.png'))

    plt.clf()
    partial_dependence_plot(
        df,
        'params_cnn_out',
        ylabel='macro F1',
        title='The partial '
        'dependence plot of cnn output channel number of each layer',
        xlabel='cnn output channel number of each layer')
    plt.savefig(os.path.join(output_root, 'cnn_out_pdp.png'))

    plt.clf()
    partial_dependence_plot(df,
                            'params_kernel_size',
                            ylabel='macro F1',
                            title='The partial '
                            'dependence plot of cnn kernel size',
                            xlabel='cnn kernel size')
    plt.savefig(os.path.join(output_root, 'kernel_size_pdp.png'))

    plt.clf()
    partial_dependence_plot(
        df,
        'params_rnn_hidden',
        ylabel='macro F1',
        title='The partial '
        'dependence plot of size of each RNN layer (per direction)',
        xlabel='hidden size of each RNN layer (per direction)')
    plt.savefig(os.path.join(output_root, 'rnn_hidden_pdp.png'))

    plt.clf()
    partial_dependence_plot(df,
                            'params_rnn_num',
                            ylabel='macro F1',
                            title='The partial dependence '
                            'plot of number of RNN layer',
                            xlabel='number of RNN layer')
    plt.savefig(os.path.join(output_root, 'rnn_num_pdp.png'))

    plt.clf()
    plot_optim_history(study)
    plt.savefig(os.path.join(output_root, 'optim_history.png'))

    with open(os.path.join(output_root, 'best_trial.txt'), 'w') as fp:
        fp.write(str(study.best_trial))

    with open(os.path.join(output_root, 'trial_number.txt'), 'w') as fp:
        fp.write(str(len(df)))


def main(optuna_trained_root, output_root):
    study = optuna.load_study(
        "seq_ann", 'sqlite:///{}/trials.db'.format(optuna_trained_root))
    save_optuna_image(study, output_root)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--optuna_trained_root", required=True)
    parser.add_argument("-o", "--output_root", required=True)

    args = parser.parse_args()
    setting = vars(args)

    main(**setting)
