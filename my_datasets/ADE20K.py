"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data.sampler import Sampler, BatchSampler
import torch.utils.data as data

class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # # random mirror
        # if random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)



        crop_size = self.crop_size
        short_size = crop_size
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # # pad crop
        # if short_size < crop_size:
        #     padh = crop_size - oh if oh < crop_size else 0
        #     padw = crop_size - ow if ow < crop_size else 0
        #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def test_img_mask(img,mask):
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(mask)
    plt.show()

class ADE20KSegmentation(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'ADEChallengeData2016'
    NUM_CLASS = 150

    def __init__(self, root='../datasets/ade', split='test', mode=None, transform=None, **kwargs):
        super(ADE20KSegmentation, self).__init__(root, split, mode, transform, **kwargs)

        self.images, self.masks = _get_ade20k_pairs(root, split)

        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])


        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        mask = np.array(mask).astype('int32') - 1
        mask[mask<=0]=0
        return torch.LongTensor(mask)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ("wall", "building, edifice", "sky", "floor, flooring", "tree",
                "ceiling", "road, route", "bed", "windowpane, window", "grass",
                "cabinet", "sidewalk, pavement",
                "person, individual, someone, somebody, mortal, soul",
                "earth, ground", "door, double door", "table", "mountain, mount",
                "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",
                "chair", "car, auto, automobile, machine, motorcar",
                "water", "painting, picture", "sofa, couch, lounge", "shelf",
                "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",
                "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",
                "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",
                "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
                "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",
                "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",
                "grandstand, covered stand", "path", "stairs, steps", "runway",
                "case, display case, showcase, vitrine",
                "pool table, billiard table, snooker table", "pillow",
                "screen door, screen", "stairway, staircase", "river", "bridge, span",
                "bookcase", "blind, screen", "coffee table, cocktail table",
                "toilet, can, commode, crapper, pot, potty, stool, throne",
                "flower", "book", "hill", "bench", "countertop",
                "stove, kitchen stove, range, kitchen range, cooking stove",
                "palm, palm tree", "kitchen island",
                "computer, computing machine, computing device, data processor, "
                "electronic computer, information processing system",
                "swivel chair", "boat", "bar", "arcade machine",
                "hovel, hut, hutch, shack, shanty",
                "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
                "motorcoach, omnibus, passenger vehicle",
                "towel", "light, light source", "truck, motortruck", "tower",
                "chandelier, pendant, pendent", "awning, sunshade, sunblind",
                "streetlight, street lamp", "booth, cubicle, stall, kiosk",
                "television receiver, television, television set, tv, tv set, idiot "
                "box, boob tube, telly, goggle box",
                "airplane, aeroplane, plane", "dirt track",
                "apparel, wearing apparel, dress, clothes",
                "pole", "land, ground, soil",
                "bannister, banister, balustrade, balusters, handrail",
                "escalator, moving staircase, moving stairway",
                "ottoman, pouf, pouffe, puff, hassock",
                "bottle", "buffet, counter, sideboard",
                "poster, posting, placard, notice, bill, card",
                "stage", "van", "ship", "fountain",
                "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
                "canopy", "washer, automatic washer, washing machine",
                "plaything, toy", "swimming pool, swimming bath, natatorium",
                "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",
                "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",
                "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
                "trade name, brand name, brand, marque", "microwave, microwave oven",
                "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",
                "bicycle, bike, wheel, cycle", "lake",
                "dishwasher, dish washer, dishwashing machine",
                "screen, silver screen, projection screen",
                "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
                "traffic light, traffic signal, stoplight", "tray",
                "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
                "dustbin, trash barrel, trash bin",
                "fan", "pier, wharf, wharfage, dock", "crt screen",
                "plate", "monitor, monitoring device", "bulletin board, notice board",
                "shower", "radiator", "glass, drinking glass", "clock", "flag")


def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print('cannot find the mask:', maskpath)

    return img_paths, mask_paths

def make_data_sampler(dataset, shuffle, distributed=False):
    # if distributed:
    #     return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = data.sampler.RandomSampler(dataset)
    else:
        sampler = data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0):
    batch_sampler = data.sampler.BatchSampler(sampler, images_per_batch, drop_last=True)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.utils.data as data
    import torch.backends.cudnn as cudnn

    from torchvision import transforms

    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    batch_size = 4

    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': batch_size, 'crop_size': 480}
    train_dataset = ADE20KSegmentation("/my_data/ADE20k", split='train', mode='train', **data_kwargs)
    val_dataset = ADE20KSegmentation("/my_data/ADE20k", split='val', mode='val', **data_kwargs)
    print("train_dataset :",len(train_dataset))
    print("val_dataset:",len(val_dataset))
    iters_per_epoch = len(train_dataset) // 4
    max_iters = 100 * iters_per_epoch

    train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=False)
    train_batch_sampler = make_batch_data_sampler(train_sampler, batch_size, max_iters)
    val_sampler = make_data_sampler(val_dataset, False, False)
    val_batch_sampler = make_batch_data_sampler(val_sampler, batch_size)

    train_loader = data.DataLoader(dataset=train_dataset,batch_sampler=train_batch_sampler)
    val_loader = data.DataLoader(dataset=val_dataset,batch_sampler=val_batch_sampler)

    for images, labels,index in train_dataset:
        print(images.shape,images.type())
        print(labels.shape,labels.type())

        break
    for images, labels,index in train_dataset:
        print(images.shape, images.type())
        print(labels.shape, labels.type())
        break