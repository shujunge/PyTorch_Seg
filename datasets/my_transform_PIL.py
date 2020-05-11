import torch
import numpy as np
from torchvision import transforms

class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()

class Label_To_EveryClass:

    def __init__(self, nlabel):
        self.nlabel = nlabel

    def __call__(self, tensor):
        tensor = [(tensor == v) for v in range(self.nlabel)]
        tensor = torch.stack(tensor, dim=0).float()
        return tensor


def train_torchvision_transforms(image_size, nclasses ):

    input_transform = transforms.Compose([
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    target_transform = transforms.Compose([
        transforms.CenterCrop((image_size, image_size)),
        ToLabel(),
        Label_To_EveryClass(nclasses)])
    return input_transform, target_transform

def inference_torchvision_transforms(image_size):
    input_transform = transforms.Compose([
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return input_transform


if __name__ == "__main__":

    from PIL import Image
    import matplotlib.pyplot as plt

    def show_results(img, lab, title_name):
        plt.figure(figsize=(4, 2))

        plt.subplot(121)
        plt.imshow(img)
        plt.title("image")
        plt.subplot(122)
        plt.imshow(lab)
        plt.title("label")
        plt.tight_layout()
        plt.title(title_name)
        plt.show()

    img_path = "/data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"
    label_path =  "/data/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png"
    img = Image.open(img_path).convert('RGB')
    lab = Image.open(label_path).convert('P')
    show_results(img, lab,title_name='base')

    input_transform = transforms.Compose([transforms.CenterCrop((320, 320))])
    x = input_transform(img)
    y = input_transform(lab)
    show_results(x, y,title_name="CenterCrop")