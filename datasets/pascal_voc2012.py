import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import jpeg4py as jpeg

class ImageData(Dataset):
    """Create image dataset from folder
    """

    def __init__(self, base_dir="/home/zfw/VOCdevkit/VOC2012",
                 split='train', transform_dtype='PIL', input_transform=None, target_transform=None):

        self.imgdir = os.path.join(base_dir, "JPEGImages")
        self.labdir = os.path.join(base_dir, "SegmentationClass")
        self.transform_dtype = transform_dtype
        index_path = os.path.join(base_dir,"ImageSets/Segmentation")
        
        with open("%s/%s.txt"% (index_path, split) ,'r') as f:
            zz = [path_str.split("\n")[0] for path_str in f.readlines()]
        self.imgfiles =zz
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.split = split

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, idx):
        
        if self.split == "test":

            img_index = self.imgfiles[idx]
            # using jpeg4py instead of Image open
            image = jpeg.JPEG(os.path.join(self.imgdir, img_index + ".jpg")).decode()

            if self.transform_dtype =='PIL':

                img = Image.fromarray(image).convert('RGB')
                # img = Image.open(os.path.join(self.imgdir, img_index + ".jpg")).convert('RGB')
                if self.input_transform is not None:
                    img = self.input_transform(img)

                return img
        else:
            img_index = self.imgfiles[idx]

            # jpef4py > opencv > PIL open

            # Using jpeg4py instead of Image open
            # image = jpeg.JPEG(os.path.join(self.imgdir, img_index + ".jpg")).decode()

            image = cv2.imread(os.path.join(self.imgdir, img_index + ".jpg"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # img = Image.open(os.path.join(self.imgdir, img_index + ".jpg")).convert('RGB')
            lab = Image.open(os.path.join(self.labdir, img_index + ".png")).convert('P')

            if self.transform_dtype == 'PIL':

                img = Image.fromarray(image).convert('RGB')

                if self.input_transform is not None:
                    img = self.input_transform(img)
                if self.target_transform is not None:
                    lab = self.target_transform(lab)

                # masks = [(lab == v) for v in range(21)]
                # mask = np.stack(masks, axis=0).astype('float32')
                lab = np.array(lab).astype('int32')
                return img, lab
            else:

                lab = np.array(lab)
                if self.input_transform is not None:
                    image = self.input_transform(image)
                if self.target_transform is not None:
                    lab = self.target_transform(lab)

                # masks = [(lab == v) for v in range(21)]
                # mask = np.stack(masks, axis=0).astype('float32')
                lab = np.array(lab).astype('int32')
                return image, lab
