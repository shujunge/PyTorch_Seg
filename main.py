import warnings
warnings.filterwarnings('ignore')
import os
import torch

from torchvision import transforms
from torch.utils import data
from datasets.VocDataset import VOCSegmentation, make_batch_data_sampler, make_data_sampler

import torch.nn as nn
from utils.my_lr import WarmupPolyLR
from utils.my_trainer import training_loop
from configs.my_argparse import my_argparse, load_config
from utils.loss import MixSoftmaxCrossEntropyLoss, EncNetLoss
from utils.score import SegmentationMetric

if __name__ == "__main__":

    weight_path = "weights"
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    result_path = "results"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # hyper-parameter

    args = my_argparse()
    cfgs = load_config(args)

    os.environ["CUDA_VISIBLE_DEVICES"] =args.GPUs 

    cfgs.model_name = '%dx%d_%s_%s_stage_%s' %(cfgs.TRAIN.image_size, cfgs.TRAIN.image_size, cfgs.MODEL.backbone, cfgs.MODEL.head, cfgs.MODEL.stage)
    # cfgs.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ##判断是否有gpu



    cfgs.save_weight_path = "./weights/%s.pt" % cfgs.model_name
    cfgs.save_tranining_path = "./results/%s.csv" % cfgs.model_name
    print(cfgs)

    Model_Params = {'DeepLabV3': {'nclass': cfgs.DATASET.nclasses, 'stage':cfgs.MODEL.stage, 'backbone': cfgs.MODEL.backbone, 'pretrained_base': True },
                    'BiSeNet': {'nclass': cfgs.DATASET.nclasses, 'backbone': cfgs.MODEL.backbone, 'pretrained_base': True},
                    'OCNet': {'nclass': cfgs.DATASET.nclasses, 'oc_arch': 'pyramid', 'stage':cfgs.MODEL.stage,'backbone': cfgs.MODEL.backbone,'pretrained_base': True},
                    'ICNet': {'nclass': cfgs.DATASET.nclasses, 'backbone': cfgs.MODEL.backbone, 'pretrained_base': True},
                    'DenseASPP': {'nclass': cfgs.DATASET.nclasses, 'backbone': cfgs.MODEL.backbone, 'pretrained_base': True},
                    'PSPNet': {'nclass': cfgs.DATASET.nclasses, 'stage':cfgs.MODEL.stage,'backbone': cfgs.MODEL.backbone, 'pretrained_base': True},
                    'DANet': {'nclass': cfgs.DATASET.nclasses,  'stage':cfgs.MODEL.stage, 'backbone': cfgs.MODEL.backbone, 'pretrained_base': True},
                    'DUNet': {'nclass': cfgs.DATASET.nclasses, 'backbone': cfgs.MODEL.backbone, 'pretrained_base': True},
                    'EncNet': {'nclass': cfgs.DATASET.nclasses, 'backbone': cfgs.MODEL.backbone, 'pretrained_base': True},
                    'UNet': {'in_channels': 3, 'n_classes': cfgs.DATASET.nclasses, 'bilinear': True, 'backbone': cfgs.MODEL.backbone,
                             'pretrained_base': True, 'usehypercolumns': False},
                    }

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': 520, 'crop_size': cfgs.TRAIN.image_size ,'root': cfgs.DATASET.dataset_path} #
    train_data = VOCSegmentation(split='train', mode='train', **data_kwargs)
    val_data = VOCSegmentation(split='val', mode='val', **data_kwargs)
    iters_per_epoch = len(train_data) // (cfgs.TRAIN.batch_size)
    max_iters = iters_per_epoch

    train_sampler = make_data_sampler(train_data, shuffle = True, distributed=False)
    train_batch_sampler = make_batch_data_sampler(train_sampler, cfgs.TRAIN.batch_size, max_iters)
    val_sampler = make_data_sampler(val_data, shuffle = False, distributed = False)
    val_batch_sampler = make_batch_data_sampler(val_sampler, cfgs.TRAIN.batch_size)

    train_Dataloader = data.DataLoader(dataset=train_data,
                                   batch_sampler=train_batch_sampler,
                                   num_workers=4,
                                   pin_memory=True)
    val_Dataloader = data.DataLoader(dataset=val_data,
                                 batch_sampler=val_batch_sampler,
                                 num_workers=4,
                                 pin_memory=True)

    print("train_dataset:", len(train_data), train_data[0][0].shape, train_data[0][1].shape)
    print("val_dataset:", len(val_data))

    model = eval(cfgs.MODEL.head)(**Model_Params[cfgs.MODEL.head]).cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    #model.load_state_dict(torch.load('weights/EfficientNet_B4_UNet_0.8055.pt'))
    optimizer = torch.optim.Adam(model.parameters(), lr= cfgs.TRAIN.lr)

    #loss_fn = nn.BCELoss()
    # loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    if cfgs.MODEL.head =='ENcNet':
        loss_fn = EncNetLoss(nclass= cfgs.DATASET.nclasses, ignore_index=-1)
    # elif cfgs.MODEL.head =='ENcNet':
        # loss_fn = ICNetLoss(nclass=cfgs.DATASET.nclasses, ignore_index=-1)
    else:
        loss_fn  = MixSoftmaxCrossEntropyLoss(aux=False, aux_weight=False, ignore_index=-1) # [nn.BCEWithLogitsLoss(), DiceLoss()]


    # lr scheduling
    iters_per_epoch = len(train_data) // (cfgs.TRAIN.batch_size)
    lr_scheduler = WarmupPolyLR(optimizer,
                                max_iters= (cfgs.TRAIN.epochs+10) * iters_per_epoch,
                                power=0.99,
                                warmup_factor=1.0 / 3,
                                warmup_iters=0,
                                warmup_method='linear')

    cfgs.train_metric = SegmentationMetric(nclass=cfgs.DATASET.nclasses)
    cfgs.val_metric = SegmentationMetric(nclass=cfgs.DATASET.nclasses)

    training_loop(cfgs, optimizer, lr_scheduler, model, loss_fn, train_Dataloader, val_Dataloader)

