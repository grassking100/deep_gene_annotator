import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import get_file_name,create_folder


def main(data_root, saved_root, save_command_table_path,
         model_config_path, executor_config_path,
         appended_command=None,model_weights_root=None):

    MAIN_PATH = os.path.dirname(__file__)+'/train_model.py'
    COMMAND = "python3 " + MAIN_PATH + " -m {} -e {} -t {} -v {} -s {}"
    paths = [MAIN_PATH, model_config_path, executor_config_path]

    for path in paths:
        if not os.path.exists(path):
            raise Exception("{} is not exists".format(path))

    data_usage_path = os.path.join(data_root, 'split_table.csv')
    data_usage = pd.read_csv(data_usage_path, comment='#').to_dict('record')
    create_folder(saved_root)
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
            if model_weights_root is not None:
                model_weights_path = os.path.join(model_weights_root,name,'checkpoint/best_model.pth')
                command += " --model_weights_path " + model_weights_path
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
    parser.add_argument("--model_weights_root", type=str)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)