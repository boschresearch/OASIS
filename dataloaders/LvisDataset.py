'''
A dataset class for LVIS making heavy use of the lvis-api (https://github.com/lvis-dataset/lvis-api)

# Installation of the lvis-api: Execute both commands
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' 
pip install lvis
'''

import os
from matplotlib import pyplot as plt
from PIL import Image
import random
import numpy as np
import argparse

import torch
from torchvision import transforms as TR
from torchvision.utils import save_image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from lvis import LVIS  # install from https://github.com/lvis-dataset/lvis-api
from lvis.colormap import colormap

def get_fixed_colormap(num_classes):
    # This implements a simple random number generator.
    # With this function, the colormap values
    # are independent of any library(versions).
    def xorshift(seed):
        seed ^= seed << 13
        seed ^= seed >> 17
        seed ^= seed << 5
        # to uint32
        seed %= int("ffffffff", 16)
        return seed

    seed = 1234
    colormap = torch.zeros(num_classes*3)
    for i in range(num_classes*3):
        seed = xorshift(seed)
        colormap[i] = seed % 10000
    colormap = colormap.reshape(num_classes, 3)/10000
    return colormap


class LvisDataset(Dataset):

    def __init__(self, opt, for_metrics=False):
        super().__init__()

        if opt.phase=="train":
            self.load_size = 286
            self.path_to_train_images = os.path.join(opt.dataroot, 'train2017')
            self.path_to_val_images = os.path.join(opt.dataroot, 'val2017')
            path_to_annotations = os.path.join(opt.dataroot, 'lvis_v1_train.json')

        elif opt.phase == "test" or for_metrics:
            self.load_size = 256 
            self.path_to_train_images = os.path.join(opt.dataroot, 'train2017')
            self.path_to_val_images = os.path.join(opt.dataroot, 'val2017')
            path_to_annotations = os.path.join(opt.dataroot, 'lvis_v1_val.json')

        
        
        opt.label_nc = 182
        opt.contain_dontcare_label = True
        opt.semantic_nc = 183 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0
        opt.crop_size = 256
        
        self.for_metrics = for_metrics 
        self.no_flip = opt.no_flip
        self.phase = opt.phase
        self.crop_size = opt.crop_size 

        ##################################
        # Load dataset information
        ##################################
        print('Loading annotations file',
              path_to_annotations, ', one moment ...')

        # This is where train val difference is manifested:
        self.lvis_dset = LVIS(path_to_annotations)

        # get lists of images and categories
        self.img_ids = self.lvis_dset.get_img_ids()
        self.images = self.lvis_dset.load_imgs(self.img_ids)

        # cat ids start at 1, meaning 0 is used as ignore
        # class of the label map in the getitem function
        self.cat_ids = self.lvis_dset.get_cat_ids()
        self.categories = self.lvis_dset.load_cats(self.cat_ids)
        self.num_classes = len(self.cat_ids)  # 1203
        # Examples
        # self.categories[0]: name = aerosol_can, label_id = 1
        # self.categories[-1]: name = zucchini, label_id = 1203

        # Use custom colormap
        # self.colormap = get_fixed_colormap(self.num_classes+1)
        # self.colormap[0,:] = 0

        # Use official colormap, where 80 colors are given to 1203 classes
        color_list = torch.from_numpy(colormap(rgb=True) / 255)
        self.colormap = torch.zeros(self.num_classes+1, 3)
        for idx in range(self.num_classes+1):
            self.colormap[idx, :] = color_list[idx % len(color_list), 0:3]

    def __len__(self):
        return len(self.images)  

    def __getitem__(self, idx):

        # pick a random image
        image_id = self.images[idx]['id']
        # Load image from disk
        coco_url = self.images[idx]['coco_url']
        image_filename = coco_url.split('/')[-1]
        #image_filename = str(image_id).zfill(12) + '.jpg'

        # example of coc_url 'blabla/train/image_xyz64.jpg'
        # example of coc_url 'blabla/val/image_xyz64.jpg'
        if 'train' in coco_url: 
            image_path = os.path.join(
                self.path_to_train_images, image_filename)
        elif 'val' in coco_url:
            image_path = os.path.join(self.path_to_val_images, image_filename)

        image = Image.open(image_path).convert('RGB')

        # Load annotations for selected image
        ann_id = self.lvis_dset.get_ann_ids(img_ids=[image_id])
        ann = self.lvis_dset.load_anns(ann_id)

        # The annotations are a list of objects.
        # For each object, we extract the object
        # mask. The object masks are then
        # combined into a label map:
        if len(ann) == 0:
            # there are un-annotated images
            label_map = torch.zeros(
                (self.load_size, self.load_size), dtype=torch.long)
        else:
            for i, single_object in enumerate(ann):
                cat_id = single_object['category_id']
                mask = self.lvis_dset.ann_to_mask(single_object)
                mask = torch.from_numpy(mask).long()

                if i == 0:
                    size = mask.size()
                    label_map = torch.zeros(size, dtype=torch.long)
                label_map = cat_id*mask + label_map*(1-mask)

        label_map = label_map.unsqueeze(0)
        image, label_map = self.transforms(image, label_map)

        return image, label_map

    def transforms(self, image, label):

        # resize
        new_width, new_height = (self.load_size, self.load_size)
        image = TR.functional.resize(
            image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(
            label, (new_width, new_height), Image.NEAREST)
        label = label.squeeze(0).long()

        # crop
        crop_x = random.randint(0, np.maximum(0, new_width - self.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.crop_size))
        image = image.crop(
            (crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        label = label[crop_x:crop_x + self.crop_size,
                      crop_y:crop_y + self.crop_size]

        # flip
        if not (self.phase == "test" or self.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)

        # to tensor
        image = TR.functional.to_tensor(image)

        # normalize
        image = TR.functional.normalize(
            image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        return image, label

    def colorize_labels(self, label_map):
        B, H, W = label_map.size()
        colored_label_map = torch.zeros((B, H, W, 3))
        unique_labels = label_map.unique()
        for lbl in unique_labels:
            colored_label_map[label_map == int(
                lbl), :] = self.colormap[int(lbl)].view(1, 1, 1, 3)
        colored_label_map = colored_label_map.permute(0, 3, 1, 2).contiguous()
        return colored_label_map


class LVIS_TrainSet(LvisDataset):
    def __init__(self, opt):
        print('Using LVIS version 1.0')

        opt.load_size = 286
        opt.crop_size = 256
        path_to_train_images = os.path.join(opt.dataroot, 'train2017')
        path_to_val_images = os.path.join(opt.dataroot, 'val2017')
        path_to_annotations = os.path.join(opt.dataroot, 'lvis_v1_train.json')

        super().__init__(path_to_train_images, path_to_val_images,
                         path_to_annotations, 'train', opt.load_size, opt.crop_size, opt.no_flip)


class LVIS_ValSet(LvisDataset):
    def __init__(self, opt):
        print('Using LVIS version 1.0')

        opt.load_size = 256
        opt.crop_size = 256
        path_to_train_images = os.path.join(opt.dataroot, 'train2017')
        path_to_val_images = os.path.join(opt.dataroot, 'val2017')
        path_to_annotations = os.path.join(opt.dataroot, 'lvis_v1_val.json')

        super().__init__(path_to_train_images, path_to_val_images,
                         path_to_annotations, 'test', opt.load_size, opt.crop_size, opt.no_flip)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,
                        default='/fs/scratch/rng_cr_bcai_dl/OpenData/LVIS_v1/')
    parser.add_argument('--load_size', type=int, default=286)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--no_flip', action='store_true',
                        help='if specified, do not flip the images for data argumentation')
    opt = parser.parse_args()

    #dataset = LVIS_TrainSet(opt)
    dataset = LVIS_ValSet(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # Demonstrate looping over dataloader
    for image, label_map in dataloader:
        print('image:', image.size())
        print('label_map:', label_map.size())
        break

    # Save output to current working dir
    colored_labels = dataset.colorize_labels(label_map)
    save_img = torch.cat(
        ((image-image.min())/(image.max()-image.min()), colored_labels), dim=0)
    save_image(save_img, 'lvis_example_batch.jpg', nrow=opt.batch_size)
 