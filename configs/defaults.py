"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()


# Total mini-batch size.
_C.TRAIN.batch_size = 28

_C.TRAIN.lr = 1e-4

_C.TRAIN.weight_path = './'

_C.TRAIN.epochs = 100

_C.TRAIN.earying_step = 15
# ---------------------------------------------------------------------------- #
# MODEL options.
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()

_C.MODEL.head = 'DeepLabV3'

_C.MODEL.backbone = 'resnest101'

_C.MODEL.stage = 'c3'

# ---------------------------------------------------------------------------- #
# DATASET options.
# ---------------------------------------------------------------------------- #
_C.DATASET = CfgNode()

_C.DATASET.name = 'VOC2012'

_C.DATASET.dataset_path = "/home/zfw/VOCdevkit"

_C.DATASET.nclasses = 21

_C.DATASET.image_size = 352


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()