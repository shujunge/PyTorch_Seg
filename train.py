import time
import datetime
import shutil
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from my_datasets.ADE20K import *
from models.pspnet import PSPNet
from loss import dice_loss
from lr_scheduler import WarmupPolyLR
from score import SegmentationMetric
from tqdm import tqdm


class Trainer(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8
        self.base_size = 520
        self.crop_size = 480
        self.distributed = False
        self.resume = True
        self.model_name = "PSPNet_resnet50"
        self.weight_path = "./checkpoint/%s.pt"%self.model_name

        self.use_ohem = False
        self.aux = False
        self.aux_weight = 0.4
        self.epochs = 100
        self.val_epoch = 1
        self.save_epoch = 5
        self.lr = 0.008
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.warmup_factor = 1.0 / 3
        self.warmup_iters = 0
        self.warmup_method = 'linear'
        self.dir = "/home/zfw/semantic_segmentation/ADEChallengeData2016"
        # self.dir = "/my_data/ADE20k" #
        self.best_pred = -2

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': self.base_size, 'crop_size': self.crop_size}
        train_dataset = ADE20KSegmentation(self.dir, split='train', mode='train', **data_kwargs)
        val_dataset = ADE20KSegmentation(self.dir, split='val', mode='val', **data_kwargs)

        self.train_loader = data.DataLoader(dataset = train_dataset, batch_size= self.batch_size)
        self.val_loader = data.DataLoader(dataset = val_dataset, batch_size= self.batch_size)
        self.iters_per_epoch = len(train_dataset) // (self.batch_size)
        self.model = PSPNet(150, backbone='resnet50', pretrained_base=True).to(self.device)


        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)


        print("Starting training %s model!"%self.model_name)
        print("device:",self.device,torch.cuda.device_count())
        print("model parameters device:",next(self.model.parameters()).is_cuda)

        # resume checkpoint if needed
        if self.resume:
            print('Resuming training, loading {}...'.format(self.weight_path))
            self.model.load_state_dict(torch.load(self.weight_path, map_location=lambda storage, loc: storage))


        # create criterion
        # self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion = dice_loss

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weight_decay)


        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters =  self.epochs*self.iters_per_epoch,
                                         power = 0.9,
                                         warmup_factor = self.warmup_factor,
                                         warmup_iters = self.warmup_iters,
                                         warmup_method = self.warmup_method)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):

        start_time = time.time()
        max_iters = self.epochs*self.iters_per_epoch

        print('Start training, Total Epochs: {:d}  Total Iterations {:d}'.format(self.epochs, max_iters))

        train_history = []
        if self.resume:
            pf = pd.read_csv("./checkpoint/%s.csv"%self.model_name)
            train_history = pf.values.tolist()

        for epoch in range(self.epochs):

            self.model.train()
            self.metric.reset()
            for images, targets, _ in tqdm(self.train_loader):
                if images.size(0)<=2:
                    break
                self.lr_scheduler.step()
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                losses = self.criterion(outputs, targets).to(self.device)

                # evaluation training metrics
                self.metric.update(outputs, targets)
                train_pixAcc, train_mIoU = self.metric.get()

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

            val_pixAcc, val_mIoU, val_loss = self.validation()
            new_pred = (val_pixAcc + val_mIoU) / 2
            if new_pred > self.best_pred:
                self.best_pred = new_pred
                torch.save(self.model.state_dict(), self.weight_path)

            print("\nepoch:{:d}/{:d}||Lr:{:.6f}||Loss:{:.4f}||val_loss:{:.4f}||val_pixAcc:{:.4f}||val_mIoU:{:.4f}".
                  format(epoch, self.epochs, self.optimizer.param_groups[0]['lr'],losses.item(),val_loss,val_pixAcc, val_mIoU))

            # write training history into csv file!
            train_history.append([losses.item(), val_loss, val_pixAcc, train_pixAcc, val_mIoU, train_mIoU])
            x = pd.DataFrame(train_history)
            x.columns= ['Loss', 'val_loss', 'val_pixAcc', 'train_pixAcc', 'val_mIoU', 'train_mIoU']
            x.to_csv("./checkpoint/%s.csv"%self.model_name, index=False)

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        print(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0

        self.metric.reset()
        model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()

        val_losses = []
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs, target)
            pixAcc, mIoU = self.metric.get()
            val_losses.append(self.criterion(outputs, target).item())

        return  pixAcc, mIoU, np.mean(val_losses)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':

    if torch.cuda.is_available():
        cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
    torch.cuda.empty_cache()
