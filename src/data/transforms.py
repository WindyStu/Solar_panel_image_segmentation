import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=(128, 160)):
    return A.Compose([
        # === 保持batch兼容的空间变换 ===
        # A.OneOf([
        #     A.HorizontalFlip(p=0.3),
        #     A.VerticalFlip(p=0),  # 垂直翻转设为0，仅占位用
        #     A.Rotate(limit=5, p=0.3),  # 小角度旋转
        # ], p=0.8),  # 80%概率应用其中一种

        # # === 光伏板特需增强 ===
        # A.CoarseDropout(
        #     max_holes=5,
        #     max_height=image_size[0] // 10,  # 最大遮挡为1/10图像高度
        #     max_width=image_size[1] // 10,
        #     fill_value=0,  # 红外图像遮挡表现为低值
        #     p=0.5
        # ),
        #
        # # === 原有图像变换 ===
        # A.RandomBrightnessContrast(
        #     brightness_limit=(-0.1, 0.1),
        #     contrast_limit=(-0.1, 0.1),
        #     p=0.3
        # ),
        # A.GaussNoise(var_limit=(5, 15), mean=0, p=0.2),

        # === 标准化 ===
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

def get_val_transforms(image_size=(128, 160)):
    return A.Compose([
        # A.PadIfNeeded(
        #     min_height=image_size[0],
        #     min_width=image_size[1],
        #     border_mode=cv2.BORDER_CONSTANT,
        #     value=0  # 红外背景填充
        # ),
        # A.CenterCrop(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})