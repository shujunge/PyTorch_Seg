from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import os
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


def get_index(path):
    """
    get the length of index for voc2012 dataset.
    path: the index of train,val or test path
    """
    with open(path,'r') as f:
        zz = f.readlines()
    return [index.split("\n")[0] for index in zz]


def show_examples(images_base, labels_base, index_list, output_path):
    results= []
    for index in tqdm(index_list):
        img = cv2.imread(os.path.join(images_base, index+".jpg"))
        # lab = cv2.imread(os.path.join(labels_base, index+".png"), 0)
        lab  = np.array(Image.open(os.path.join(labels_base, index+".png")).convert('P'))
        results+= np.unique(lab).tolist()
        #
        # plt.figure(figsize=(4,2))
        # plt.subplot(121)
        # plt.imshow(img)
        # plt.title("images")
        # plt.subplot(122)
        # plt.imshow(lab)
        # plt.title('label')
        # plt.tight_layout()
        # plt.savefig("%s/visual_%s.png"%(output_path, index), dpi=300)
        # plt.show()

    return list(set(results))


def get_info(label_dir):
    label_path = glob("%s/*" % label_dir)
    total_area = []
    total_number = []

    for label_name in tqdm(label_path):
        lab = np.array(Image.open(label_name).convert('P'))
        # print(lab.shape)
        masks = [(lab == v) for v in range(21)]
        # get each class area of images
        zz = np.mean(masks, axis =(1, 2))
        total_area.append(zz.copy())
        # get exist class of images
        zz[zz > 0] = 1
        total_number.append(zz)

    print(np.sum(total_number, axis=0))
    print(np.sum(total_area, axis=0))


if __name__=="__main__":

    import shutil
    output_dir = "visual_results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    index_dir = '/data/VOCdevkit/VOC2012/ImageSets/Segmentation'
    imge_dir = "/data/VOCdevkit/VOC2012/JPEGImages"
    label_dir = "/data/VOCdevkit/VOC2012/SegmentationClass"
    print("train_index:", len(get_index( os.path.join(index_dir, "train.txt") ) ) ) # 1464
    print("val_index:", len( get_index( os.path.join(index_dir, "val.txt") ) ) ) # 1449
    print("test_index:", len( get_index( os.path.join(index_dir, "test.txt") ) ) ) #1456

    train_results= show_examples(imge_dir, label_dir, get_index(os.path.join(index_dir, "train.txt")), output_dir)
    train_results.sort()
    print("train label:", len(train_results), train_results)
    get_info(label_dir)


"""
train label: 20 [0, 14, 19, 33, 37, 38, 52, 57, 72, 75, 89, 94, 108, 112, 113, 128, 132, 147, 150, 220]

number of each class:
[2903.  178.  144.  208.  150.  183.  152.  255.  250.  271.  135.  157. 249.  147.  157.  888.  167.  120.  183.  167.  157.]

are of each class:
[2019.413   21.703    8.608   23.93    16.14    19.298   49.044   40.491
   68.606   27.83    28.275   33.941   51.712   27.909   30.196  139.84
   16.282   22.923   39.572   44.975   22.053]
"""