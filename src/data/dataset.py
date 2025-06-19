import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SolarPanelDataset(Dataset):
    def __init__(self, image_dir, transform=None, mode='train'):
        """
        Args:
            image_dir (str): 包含image和mask的目录
            transform (callable): 数据增强
            mode (str): 'train' or 'test'
        """
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.images = sorted([f for f in os.listdir(image_dir) if f.startswith('image')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = np.array(Image.open(img_path)).astype(np.float32) / 255.0

        if self.mode == 'train':
            mask_path = os.path.join(self.image_dir, self.images[idx].replace('image', 'mask'))
            mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
            mask = np.where(mask > 0.5, 1.0, 0.0)  # 二值化

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                return augmented['image'], augmented['mask'].float()
            return ToTensorV2()(image=image, mask=mask)['image'], torch.from_numpy(mask).float()

        else:
            if self.transform:
                augmented = self.transform(image=image)
                return augmented['image'], self.images[idx]
            return ToTensorV2()(image=image)['image'], self.images[idx]

class SolarPanelSubset(Dataset):
    """根据文件名列表创建子数据集"""
    def __init__(self, root_dir, file_list, transform=None, mode='train'):
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = np.array(Image.open(img_path)).astype(np.float32) / 255.0

        if self.mode == 'train':
            # mask_path = os.path.join(self.image_dir, self.images[idx].replace('image', 'mask'))
            mask_path = os.path.join(self.root_dir, img_name.replace('image', 'mask'))
            mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
            mask = np.where(mask > 0.5, 1.0, 0.0)  # 二值化

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                return augmented['image'], augmented['mask'].float()
            return ToTensorV2()(image=image, mask=mask)['image'], torch.from_numpy(mask).float()

        else:
            if self.transform:
                augmented = self.transform(image=image)
                return augmented['image'], self.images[idx]
            return ToTensorV2()(image=image)['image'], self.images[idx]
