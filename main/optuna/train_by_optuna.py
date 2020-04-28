import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/..")
from sequence_annotation.genome_handler.select_data import load_data
from sequence_annotation.process.optuna import OptunaTrainer
from sequence_annotation.utils.utils import write_json, copy_path, create_folder, read_json
from main.model_executor_creator import ModelExecutorCreator
from main.utils import backend_deterministic
from main.deep_learning.train_model import train


def main(saved_root, train_data_path, val_data_path,
         epoch, batch_size, n_startup_trials, n_trials, is_maximize,
         clip_grad_norm=None, has_cnn=False, grad_norm_type=None,
         use_lr_scheduler=False, **kwargs):

    backend_deterministic(False)
    creator = ModelExecutorCreator(clip_grad_norm=clip_grad_norm,
                                   grad_norm_type=grad_norm_type,
                                   has_cnn=has_cnn,
                                   use_lr_scheduler=use_lr_scheduler)

    space_size_path = os.path.join(saved_root, 'space_size.json')
    space_size = {'space size': creator.space_size}
    if os.path.exists(space_size_path):
        existed = read_json(space_size_path)
        if space_size != existed:
            raise Exception(
                "The {} is not same as previous one".format(space_size_path))
    else:
        write_json(space_size, space_size_path)

    # Load, parse and save data
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)
    copied_paths = [train_data_path, val_data_path]
    source_backup_path = os.path.join(saved_root, 'source')
    create_folder(source_backup_path)
    for path in copied_paths:
        copy_path(source_backup_path, path)

    if is_maximize:
        monitor_target = 'val_macro_F1'
    else:
        monitor_target = 'val_loss'

    trainer = OptunaTrainer(train, saved_root, creator, train_data, val_data, epoch, batch_size,
                            monitor_target=monitor_target, is_minimize=not is_maximize,
                            **kwargs)
    trainer.optimize(n_startup_trials, n_trials)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root", required=True,
                        help="Root to save file")
    parser.add_argument("-t", "--train_data_path", required=True,
                        help="Path of training data")
    parser.add_argument("-v", "--val_data_path", required=True,
                        help="Path of validation data")
    parser.add_argument("-g", "--gpu_id", default=0,
                        help="GPU to used", type=int)
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-n", "--n_trials", type=int, default=None)
    parser.add_argument("-i", "--n_startup_trials", type=int, default=None)
    parser.add_argument("--is_maximize", action='store_true')
    parser.add_argument("--by_grid_search", action='store_true')
    parser.add_argument("--trial_start_number", type=int, default=0)
    parser.add_argument("--augment_up_max", type=int, default=0)
    parser.add_argument("--augment_down_max", type=int, default=0)
    parser.add_argument("--has_cnn", action='store_true')
    parser.add_argument("--clip_grad_norm", type=float, default=0)
    parser.add_argument("--grad_norm_type", type=str, default=None)
    parser.add_argument("--save_distribution", action='store_true')
    parser.add_argument("--discard_ratio_min", type=float, default=0)
    parser.add_argument("--discard_ratio_max", type=float, default=0)
    parser.add_argument("--use_lr_scheduler", action='store_true')

    args = parser.parse_args()
    config = dict(vars(args))
    # Create folder
    if not os.path.exists(args.saved_root):
        os.mkdir(args.saved_root)
    # Save setting
    setting_path = os.path.join(args.saved_root, "optuna_setting.json")

    if os.path.exists(setting_path):
        existed = read_json(setting_path)
        config_ = dict(config)
        del existed['gpu_id']
        del existed['n_trials']
        del config_['gpu_id']
        del config_['n_trials']
        if config_ != existed:
            print(config)
            print(existed)
            raise Exception(
                "The {} is not same as previous one".format(setting_path))
    else:
        write_json(config, setting_path)

    del config['gpu_id']
    with torch.cuda.device(args.gpu_id):
        main(**config)
