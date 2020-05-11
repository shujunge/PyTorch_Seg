
from torch.utils.data import DataLoader
from datasets.pascal_voc2012 import ImageData
from tqdm import tqdm
from time import time
from datasets.my_transform_PIL import train_torchvision_transforms,inference_torchvision_transforms


def test_traindataset_shape():

    base_dir = "/data/VOCdevkit/VOC2012"
    batch_size = 2
    image_size = 320
    nclasses = 21

    input_transform, target_transform = train_torchvision_transforms(image_size, nclasses)

    train_data = ImageData(base_dir, split='train', input_transform=input_transform, target_transform=target_transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for image, label in train_dataloader:
        assert image.size() == (batch_size, 3, image_size, image_size)
        assert label.size() == (batch_size, nclasses, image_size, image_size)
        break

def test_valdataset_shape():

    base_dir = "/data/VOCdevkit/VOC2012"
    batch_size = 2
    image_size = 320
    nclasses = 21

    input_transform, target_transform = train_torchvision_transforms(image_size, nclasses)

    val_data = ImageData(base_dir, split='val', input_transform=input_transform,target_transform= target_transform)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for image, label in val_dataloader:
        assert image.size() == (batch_size, 3, image_size, image_size)
        assert label.size() == (batch_size, nclasses, image_size, image_size)
        break

def test_testdataset_shape():

    base_dir = "/data/VOCdevkit/VOC2012"
    batch_size = 2
    image_size = 320
    nclasses = 21

    input_transform = inference_torchvision_transforms(image_size)

    test_data = ImageData(base_dir, split='test', input_transform=input_transform)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    for image in test_dataloader:
        assert image.size() == (batch_size, 3, image_size, image_size)
        break

def test_traindataset_time():

    base_dir = "/data/VOCdevkit/VOC2012"
    batch_size = 2
    image_size = 320
    nclasses = 21

    input_transform, target_transform = train_torchvision_transforms(image_size, nclasses)

    train_data = ImageData(base_dir, split='train', input_transform=input_transform, target_transform=target_transform)
    print("train_dataloader times testing!")
    for num_worker in [2]:
        start_time = time()
        train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_worker, shuffle=True)
        for image, label in tqdm(train_dataloader):
            pass
        print("when numb_workers are %d, train_dataloader costs %s s."%(num_worker, time() - start_time))

    """
    simple dataloader: when numb_workers are 2, train_dataloader costs 25.996397256851196 s.
    jpeg decoder when numb_workers are 2, train_dataloader costs 20.658571004867554 s.
    """