import argparse

def my_argparse():

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation for VOC2012."
    )
    parser.add_argument(
        "--DATASET_PATH",
        default="/home/zfw/VOCdevkit/VOC2012",
        help="path to VCO2012 dataset ",
        type=str,
    )
    parser.add_argument(
        "--backbone",
        default="resnet101",
        choices=['resnet50','resnet101',  'EfficientNet_B4','resnest50','resnest101','RegNet200'],
        help="config the backbone of model",
        type=str,
    )

    parser.add_argument(
        "--head",
        default='DeepLabV3',
        choices=['DeepLabV3', 'UNet','BiSeNet','OCNet','ICNet'],
        help="head to Segmentation",
        type=str
    )
    parser.add_argument(
        "--weight_path",
        default=None,
        help="config path of model weight",
        type=str
    )

    parser.add_argument(
        "--GPUs",
        default='8,9',
        help="choice gpus for training",
        type=str
    )

    parser.add_argument(
        "--image_size",
        help="the size of image",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help="the batch size for model training",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        help="the epochs for model training",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--lr",
        help="set learning rate for model training",
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--earying_step",
        help="the earying_step for model training",
        default=15,
        type=int,
    )

    args = parser.parse_args()

    return args

if __name__=="__main__":
    arg = my_argparse()
    print(arg)
    print(arg.backbone, arg.head, arg.image_size)


# cfg.merge_from_file(args.cfg)
# cfg.merge_from_list(args.opts)
# # cfg.freeze()
#
# logger = setup_logger(distributed_rank=0)  # TODO
# logger.info("Loaded configuration file {}".format(args.cfg))
# logger.info("Running with config:\n{}".format(cfg))
#
# # absolute paths of model weights
# cfg.MODEL.weights_encoder = os.path.join(
#     cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
# cfg.MODEL.weights_decoder = os.path.join(
#     cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
# assert os.path.exists(cfg.MODEL.weights_encoder) and \
#        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
#
# if not os.path.isdir(os.path.join(cfg.DIR, "result")):
#     os.makedirs(os.path.join(cfg.DIR, "result"))
#
# main(cfg, args.gpu)