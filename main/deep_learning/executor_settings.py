import sys,os
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_json,get_time_str


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-o","--output_path",help="Path to save settings",required=True)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--loss_type",type=str)
    parser.add_argument("--inference_type",type=str)
    #
    parser.add_argument("--set_lr_scheduler",action="store_true")
    parser.add_argument("--lr_scheduler_factor",type=float)
    parser.add_argument("--lr_scheduler_patience",type=float)
    #
    parser.add_argument("--set_grad_clipper",action="store_true")
    parser.add_argument("--clip_grad_value",type=float)
    parser.add_argument("--clip_grad_norm",type=float)
    parser.add_argument("--grad_norm_type",type=float)
    #
    parser.add_argument("--optim_type",type=str)
    parser.add_argument("--lr",type=float)
    #
    parser.add_argument("--aug_up_max",type=int)
    parser.add_argument("--aug_down_max",type=int)
    parser.add_argument("--discard_ratio_min",type=float)
    parser.add_argument("--discard_ratio_max",type=float)
    parser.add_argument("--both_discard_order",action="store_true")
    parser.add_argument("--concat",action="store_true")
    parser.add_argument("--drop_last",action="store_true")
    #     
    args = parser.parse_args()
    kwargs = vars(args)
    settings = {}
    settings['generated_time'] = get_time_str()
    settings['lr_scheduler_kwargs'] = {}
    settings['grad_clipper_kwargs'] = {}
    settings['optimizer_kwargs'] = {}
    settings['train_data_generator_kwargs'] = {}
    basic_types = ['batch_size','loss_type','inference_type','set_lr_scheduler','set_grad_clipper']
    lr_scheduler_keywords = ['lr_scheduler_factor','lr_scheduler_patience']
    grad_clipper_keywords = ['clip_grad_value','clip_grad_norm','grad_norm_type']
    optimizer_keywords = ['optim_type','lr']
    train_data_generator_keywords = ['aug_up_max','aug_down_max','discard_ratio_min','discard_ratio_max',
                                     'both_discard_order','drop_last','concat']
    for type_ in basic_types:
        settings[type_] = kwargs[type_]
        
    for type_ in lr_scheduler_keywords:
        settings['lr_scheduler_kwargs'][type_] = kwargs[type_]
        
    for type_ in grad_clipper_keywords:
        settings['grad_clipper_kwargs'][type_] = kwargs[type_]
        
    for type_ in optimizer_keywords:
        settings['optimizer_kwargs'][type_] = kwargs[type_]
        
    for type_ in train_data_generator_keywords:
        settings['train_data_generator_kwargs'][type_] = kwargs[type_]

    write_json(settings,args.output_path)
