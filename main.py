import warnings
warnings.filterwarnings('ignore')
import os
#os.environ["CUDA_VISIBLE_DEVICES"] ="8,9"
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torchvision import transforms
from torch.utils import data
from tqdm import tqdm
from datasets.VocDataset import VOCSegmentation, make_batch_data_sampler, make_data_sampler

from models.deeplabv3 import DeepLabV3
from models.unet import UNet
from models.bisenet import BiSeNet
from models.OCNet import OCNet
import torch.nn as nn
import pandas as pd
from utils.my_lr import WarmupPolyLR
from utils.my_trainer import training_loop
from utils.my_argparse import my_argparse
from utils.loss import MixSoftmaxCrossEntropyLoss
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
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] =args.GPUs 

    args.model_name = '%dx%d_%s_%s' %(args.image_size, args.image_size, args.backbone, args.head)
    args.nclasses = 21
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ##判断是否有gpu
    if torch.cuda.is_available():
        cudnn.benchmark = True

    Model_Zoos = {
        # '%s_DeepLabV3'%(args.backbone): DeepLabV3( backbone_name= args.backbone, num_classes=args.nclasses),
        # '%s_BiSeNet' % (args.backbone): BiSeNet(nclass=args.nclasses, backbone='resnest101',pretrained_base=True),
        '%s_OCNet' % (args.backbone): OCNet(nclass=args.nclasses,oc_arch='pyramid', backbone='resnest101', pretrained_base=True),
        # '%s_UNet'%(args.backbone): UNet(in_channels= 3, n_classes=args.nclasses, bilinear=True, backbone= args.backbone, pretrained_base=True, usehypercolumns=False)

        }

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': 520, 'crop_size': args.image_size ,'root': "/home/zfw/VOCdevkit"} #
    train_data = VOCSegmentation(split='train', mode='train', **data_kwargs)
    val_data = VOCSegmentation(split='val', mode='val', **data_kwargs)
    iters_per_epoch = len(train_data) // (args.batch_size)
    max_iters = iters_per_epoch

    train_sampler = make_data_sampler(train_data, shuffle = True, distributed=False)
    train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, max_iters)
    val_sampler = make_data_sampler(val_data, shuffle = False, distributed = False)
    val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

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

    model = Model_Zoos["%s_%s" % (args.backbone, args.head)].to(args.device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    #model.load_state_dict(torch.load('weights/EfficientNet_B4_UNet_0.8055.pt'))
    optimizer = torch.optim.Adam(model.parameters(), lr= args.lr)

    #loss_fn = nn.BCELoss()
    # loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    loss_fn  = MixSoftmaxCrossEntropyLoss(aux=False, aux_weight=False, ignore_index=-1) # [nn.BCEWithLogitsLoss(), DiceLoss()]


    # lr scheduling
    iters_per_epoch = len(train_data) // (args.batch_size)
    lr_scheduler = WarmupPolyLR(optimizer,
                                max_iters= (args.epochs+10) * iters_per_epoch,
                                power=0.99,
                                warmup_factor=1.0 / 3,
                                warmup_iters=0,
                                warmup_method='linear')

    args.train_metric = SegmentationMetric(nclass=args.nclasses)
    args.val_metric = SegmentationMetric(nclass=args.nclasses)

    training_loop(args, optimizer, lr_scheduler, model, loss_fn, train_Dataloader, val_Dataloader)
