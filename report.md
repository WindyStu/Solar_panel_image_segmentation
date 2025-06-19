# Lab Report: Image Segmentation for Photovoltaic Panel Detection

## Background
- **Dataset Description**
  - Training Set: 2,693 images
  - Test Set A: 336 images
  - Test Set B: 338 images (final evaluation criterion)
- **Evaluation Metrics**
  - Intersection over Union (IoU)
  - $\frac{TP}{TP+FP+FN}$
- **Image Specifications**
  - Input Image Size: 160×128 pixels
  - Image Naming Convention: image0.JPG
  - Mask Naming Convention: mask0.JPG
  - Prediction Naming Convention: pred0.JPG

## 使用命令行训练/预测，查看日志信息
```shell
python run.py --config configs/base.yaml #train
```

```shell
python src/predict.py --model_path outputs/checkpoints/best_model.pth --test_dir data/testA --output_dir submissions/22211870121
```

```shell
tensorboard --logdir=outputs/logs/{model_name}
```
## 实验过程
1. 使用 U-Net 模型


## 实验踩坑汇总
1. 数据增强没有同步到mask(如水平、垂直翻转等),浪费大部分训练时间(具体表现为val loss不如数据增强前的)
2. 补充1 准确来说 应该对mask作用位置变换而不作用像素变换 可以使用Albumentations的掩码保护操作 参考博客(https://ask.csdn.net/questions/8364948)
3. 前期一直使用val_loss作为评价指标，结果一改loss就没有模型间可比较性，最后使用val的IoU最为评价指标并使用早停减少模型无效训练时间

## 实验记录
| Model | transforms           | Loss                | IoU(testA) | IoU(val) |
| --- |----------------------|---------------------|------------|----------|
| U-Net | none                 | DiceBCELoss         | 0.9667     | -        |
| U-Net | basic+CoarseDropout+RandomBright | dice_bce+edge_loss+focal_loss | 0.9573 | -        |        |
| U-Net|HorizontalFlip| DiceBCELoss | - | 0.9721   |

## 疑点
1. 尝试过使用复杂参数量大的模型，效果很差，怀疑是因为数据集过于简单导致过拟合。
2. 接1 使用复杂的数据增强尝试防止过拟合，效果依旧很差，不懂
3. 