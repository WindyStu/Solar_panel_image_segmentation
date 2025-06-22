import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from src.data.dataset import *
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.improved_unet import LightweightUNet
from src.models.nested_unet import NestedUNet, AttU_Net, R2AttU_Net,U_Net
from src.models.unet import UNet
from src.models.losses import DiceBCELoss, SolarPanelLoss, DiceBCELoss_with_L2
from src.models.DeepLabv3 import DeepLabV3Plus
from src.utils.logger import TensorBoardLogger
from sklearn.model_selection import train_test_split
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        # 初始化模型
        # self.model = UNet(n_channels=3, n_classes=1).to(self.device)
        # self.model = NestedUNet(in_ch=3, out_ch=1).to(self.device)
        # self.model = AttU_Net(img_ch=3, output_ch=1).to(self.device)
        self.model = U_Net(in_ch=3, out_ch=1).to(self.device)
        # self.model = DeepLabV3Plus(n_classes=1).to(self.device)
        # self.model = LightweightUNet().to(self.device)
        # self.model = R2AttU_Net(in_ch=3, out_ch=1, t=1).to(self.device)
        self.model.load_state_dict(torch.load(config['model_path']))
        if self.device.type == 'cuda':
            torch.backends.cudnn.enable = True
            torch.backends.cudnn.benchmark = True

        all_images = sorted([f for f in os.listdir("data/train") if f.startswith('image')])
        train_images, val_images = train_test_split(
            all_images,
            test_size=0.2,
            random_state=42
        )

        # 数据加载
        train_dataset = SolarPanelSubset("data/train", train_images,transform=get_train_transforms())
        # train_dataset = SolarPanelSubset("data/train", train_images)
        val_dataset = SolarPanelSubset("data/train", val_images, transform=get_val_transforms())
        # val_dataset = SolarPanelSubset("data/train", val_images)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=4
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=False,  # 验证集不需要shuffle
            num_workers=4
        )

        # 优化器与损失
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config['train']['lr']),
            weight_decay=config['train']['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.criterion = DiceBCELoss_with_L2()

        # 日志记录
        self.logger = TensorBoardLogger(
            log_dir=Path(config['logging']['log_dir']) / config['model']['name'] / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    def calculate_iou(self, pred, target):
        # 将预测值进行sigmoid激活，大于0.5的置为1，小于等于0.5的置为0
        pred = (torch.sigmoid(pred) > 0.5).float()
        # 将目标值转换为浮点型
        target = target.float()
        # 如果预测值的维度为4，则去掉第一个维度
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        # 计算预测值和目标值的交集
        intersection = (pred * target).sum()
        # 计算预测值和目标值的并集
        union = (pred + target).sum() - intersection
        # 返回交集与并集的比值
        return intersection / (union + 1e-6)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_iou = 0.0
        scaler = torch.cuda.amp.GradScaler()

        for images, masks in tqdm(self.train_loader, desc="Training"):
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(self.criterion, outputs, masks)
                batch_iou = self.calculate_iou(outputs, masks)
                total_iou += batch_iou.item()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数（自动处理缩放）
            scaler.step(self.optimizer)
            scaler.update()

            running_loss += loss.item()

        return running_loss / len(self.train_loader), total_iou / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                val_loss += self.criterion(self.model, outputs, masks).item()
                batch_iou = self.calculate_iou(outputs, masks)
                # print(batch_iou)
                total_iou += batch_iou.item()

        return val_loss / len(self.val_loader), total_iou / len(self.val_loader)

    def run(self):
        best_val_loss = float('inf')
        best_iou = 0
        patience = 10
        no_improve = 0

        for epoch in range(self.config['train']['epochs']):
            train_loss, train_iou= self.train_epoch()
            val_loss , val_iou= self.validate()
            self.scheduler.step(val_loss)

            # 记录日志
            self.logger.log_scalar('Loss/train', train_loss, epoch)
            self.logger.log_scalar('Loss/val', val_loss, epoch)

            # 保存最佳模型
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(self.model.state_dict(),
            #                Path(self.config['logging']['checkpoint_dir']) / self.config['model']['name']/ 'best_model.pth')

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(),
                           Path(self.config['logging']['checkpoint_dir']) / self.config['model']['name']/ 'best_model.pth')
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"早停 @ epoch {epoch}, 最佳IoU: {best_iou:.4f}")
                    break

            print(f"Epoch {epoch + 1}/{self.config['train']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                  f"| Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}")

