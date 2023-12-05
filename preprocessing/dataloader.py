import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from augmentation import randomHueSaturationValue, randomShiftScaleRotate, randomFlip, randomRotate90x

def loader(img_path, mask_path, phase):

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  #, cv2.IMREAD_GRAYSCALE

    img = cv2.resize(img, (256,256),cv2.INTER_AREA)
    mask = cv2.resize(mask, (256,256),cv2.INTER_AREA)

    img=Image.open(img_path)
    img=img.resize((256,256), Image.ANTIALIAS)
    mask=Image.open(mask_path)
    mask=mask.resize((256,256), Image.ANTIALIAS)

    if(phase == 'train'):
       img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-30, 30),
                                     sat_shift_limit=(-5, 5),
                                     val_shift_limit=(-15, 15)
                                     )

       img, mask = randomShiftScaleRotate(img, mask,
                                         shift_limit=(-0.1, 0.1),
                                         scale_limit=(-0.1, 0.1),
                                         aspect_limit=(-0.1, 0.1),
                                         rotate_limit=(-0, 0))
       img, mask = randomFlip(img, mask)
       img, mask = randomRotate90(img, mask)

    img = np.array(img, np.float32).transpose(2,0,1)/255.0
    mask = np.array(mask, np.float32)
    mask = mask[np.newaxis,:,:]
    mask = mask/255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    return img, mask

def read_dataset(root_path, mode):
  images = []
  masks = []

  if(mode == 'Train'):
    image_root = os.path.join(root_path, 'Train/images')
    gt_root = os.path.join(root_path, 'Train/mask')
  else :
    image_root = os.path.join(root_path, 'Train/images')
    gt_root = os.path.join(root_path, 'Train/mask')

  for image_name in sorted(os.listdir(image_root)):
    image_path = os.path.join(image_root, image_name) #.split('.')[0] + '.jpg')
    images.append(image_path)
  for mask_name in sorted(os.listdir(gt_root)):
    label_path = os.path.join(gt_root, mask_name) #.split('.')[0] + '.tif')
    masks.append(label_path)
  return images, masks



class Eye_Dataset(Dataset):

    def __init__(self, root_path, phase):
        self.root = root_path
        self.phase = phase
        self.images, self.labels = read_dataset(self.root, self.phase)
        print('images: ', self.images)
        print('labels: ', self.labels)

    def __getitem__(self, index):

        img, mask = loader(self.images[index], self.labels[index], self.phase)
        img = torch.tensor(img, dtype = torch.float32)
        mask = torch.tensor(mask, dtype = torch.float32)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)