import argparse
from configs.defaults import get_cfg
def my_argparse():

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation for VOC2012."
    )
    parser.add_argument(
        "--cfg_file",
        help="Path to the config file",
        default="configs/DeepLabV3.yaml",
        type=str,
    )

    parser.add_argument(
        "--GPUs",
        default='8,9',
        help="choice gpus for training",
        type=str
    )
    # parser.add_argument(
    #     "--dataset_path",
    #     default="/home/zfw/VOCdevkit",
    #     help="path to VCO2012 dataset ",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--backbone",
    #     default="resnet101",
    #     choices=['resnet50','resnet50_v1s',
    #              'resnet101',  'resnet101_v1s','resnet152_v1s',
    #              'EfficientNet_B4','resnest50','resnest101','RegNet200', 'densenet121'],
    #     help="config the backbone of model \n Segment_base in ['resnet50','resnet50_v1s',\
    #          'resnet101',  'resnet101_v1s','resnet152_v1s', 'EfficientNet_B4','resnest50','resnest101']",
    #     type=str,
    # )
    #
    # parser.add_argument(
    #     "--head",
    #     default='DeepLabV3',
    #     choices=['DeepLabV3', 'UNet','BiSeNet','OCNet','ICNet','DenseASPP','DANet','DUNet','EncNet','PSPNet'],
    #     help="head to Segmentation",
    #     type=str
    # )
    # parser.add_argument(
    #     "--stage",
    #     default='c3',
    #     choices=['c3', 'c4'],
    #     help="the size of encoder features \n c3,c4 for [DeepLabV3,PSPNet,OCNet, DANet] ",
    #     type=str
    # )
    # parser.add_argument(
    #     "--weight_path",
    #     default=None,
    #     help="config path of model weight",
    #     type=str
    # )



    # parser.add_argument(
    #     "--image_size",
    #     help="the size of image",
    #     default=256,
    #     type=int,
    # )
    # parser.add_argument(
    #     "--batch_size",
    #     help="the batch size for model training",
    #     default=32,
    #     type=int,
    # )
    # parser.add_argument(
    #     "--epochs",
    #     help="the epochs for model training",
    #     default=100,
    #     type=int,
    # )
    # parser.add_argument(
    #     "--lr",
    #     help="set learning rate for model training",
    #     default=1e-4,
    #     type=float,
    # )
    #
    # parser.add_argument(
    #     "--earying_step",
    #     help="the earying_step for model training",
    #     default=15,
    #     type=int,
    # )

    return parser.parse_args()

def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    return cfg


