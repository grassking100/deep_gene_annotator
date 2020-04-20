import os
import sys
import pandas as pd
from argparse import ArgumentParser
ROOT = os.path.dirname(__file__) + "/.."
sys.path.append(ROOT)
from sequence_annotation.utils.utils import get_file_name


def main(data_root, saved_root, save_command_table_path,
         model_config_path, executor_config_path, appended_command=None):

    MAIN_PATH = '{}/main/train_model.py'.format(ROOT)
    COMMAND = "python3 " + MAIN_PATH + " -m {} -e {} -t {} -v {} -s {}"
    paths = [MAIN_PATH, model_config_path, executor_config_path]

    for path in paths:
        if not os.path.exists(path):
            raise Exception("{} is not exists".format(path))

    data_usage_path = os.path.join(data_root, 'data_usage_path.csv')
    data_usage = pd.read_csv(data_usage_path, comment='#').to_dict('record')

    with open(save_command_table_path, "w") as fp:
        fp.write("command\n")
        for paths in data_usage:
            training_path = os.path.join(data_root, paths['training_path'])
            validation_path = os.path.join(data_root, paths['validation_path'])
            name = get_file_name(paths['training_path']) + \
                "_" + get_file_name(paths['validation_path'])
            new_saved_root = os.path.join(saved_root, name)
            if not os.path.exists(training_path):
                raise Exception("The {} is not exists".format(training_path))
            if not os.path.exists(validation_path):
                raise Exception("The {} is not exists".format(validation_path))
            command = COMMAND.format(model_config_path, executor_config_path,
                                     training_path, validation_path, new_saved_root)
            if appended_command is not None:
                command += " " + appended_command
            command += "\n"
            fp.write(command)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_root", required=True,
                        help='The data folder generated by batch_select_data.py')
    parser.add_argument("-s", "--saved_root", required=True)
    parser.add_argument("-c", "--save_command_table_path", required=True)
    parser.add_argument("-e", "--executor_config_path", required=True,
                        help="Path of Executor config")
    parser.add_argument("-m", "--model_config_path", required=True,
                        help="Path of Model config")
    parser.add_argument("--appended_command", type=str)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
