import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5, smooth=1e-6):
        super().__init__()
        self.weight = weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B,1,H,W] 未激活的logits
            targets: [B,H,W] 二值化掩码(0/1)
        """
        # 维度处理：移除通道维度（如果target是[B,1,H,W]）
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        assert targets.dim() == 3, f"Target should be [B,H,W], got {targets.shape}"

        # 输入校验
        assert torch.all((targets == 0) | (targets == 1)), "Target must be binary (0/1)"
        targets = targets.float()
        inputs = inputs.squeeze(1)  # [B,1,H,W] -> [B,H,W]

        # 展平处理
        inputs = inputs.view(-1)  # [B*H*W]
        targets = targets.view(-1)

        # BCE计算（自动处理logits）
        bce_loss = self.bce(inputs, targets)

        # Dice计算（inputs需sigmoid）
        probas = torch.sigmoid(inputs)
        intersection = (probas * targets).sum()
        denominator = (probas.sum() + targets.sum())
        dice_loss = 1 - (2. * intersection + self.smooth) / (denominator + self.smooth)

        return self.weight * bce_loss + (1 - self.weight) * dice_loss

class DiceBCELoss_with_L2(nn.Module):
    def __init__(self,weight=0.5, smooth=1e-6):
        super().__init__()
        self.weight = weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, model, inputs, targets):
        """
        Args:
            inputs: [B,1,H,W] 未激活的logits
            targets: [B,H,W] 二值化掩码(0/1)
            :param model:
        """
        # 维度处理：移除通道维度（如果target是[B,1,H,W]）
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        assert targets.dim() == 3, f"Target should be [B,H,W], got {targets.shape}"

        # 输入校验
        assert torch.all((targets == 0) | (targets == 1)), "Target must be binary (0/1)"
        targets = targets.float()
        inputs = inputs.squeeze(1)  # [B,1,H,W] -> [B,H,W]

        # 展平处理
        inputs = inputs.view(-1)  # [B*H*W]
        targets = targets.view(-1)

        # BCE计算（自动处理logits）
        bce_loss = self.bce(inputs, targets)

        # Dice计算（inputs需sigmoid）
        probas = torch.sigmoid(inputs)
        intersection = (probas * targets).sum()
        denominator = (probas.sum() + targets.sum())
        dice_loss = 1 - (2. * intersection + self.smooth) / (denominator + self.smooth)

        l2_reg = torch.tensor(0., device=inputs.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)

        return self.weight * bce_loss + (1 - self.weight) * dice_loss + 0.0001 * l2_reg


class DynamicDiceBCELoss(nn.Module):
    def __init__(self, init_weight=0.5, smooth=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(init_weight))
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, model, inputs, targets):
        """
        Args:
            inputs: [B,1,H,W] 未激活的logits
            targets: [B,H,W] 二值化掩码(0/1)
            :param model:
        """
        # 维度处理：移除通道维度（如果target是[B,1,H,W]）
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        assert targets.dim() == 3, f"Target should be [B,H,W], got {targets.shape}"

        # 输入校验
        assert torch.all((targets == 0) | (targets == 1)), "Target must be binary (0/1)"
        targets = targets.float()
        inputs = inputs.squeeze(1)  # [B,1,H,W] -> [B,H,W]

        # 展平处理
        inputs = inputs.view(-1)  # [B*H*W]
        targets = targets.view(-1)

        # BCE计算（自动处理logits）
        bce_loss = self.bce(inputs, targets)

        # Dice计算（inputs需sigmoid）
        probas = torch.sigmoid(inputs)
        intersection = (probas * targets).sum()
        denominator = (probas.sum() + targets.sum())
        dice_loss = 1 - (2. * intersection + self.smooth) / (denominator + self.smooth)

        l2_reg = torch.tensor(0., device=inputs.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)

        return self.weight * bce_loss + (1 - self.weight) * dice_loss + 0.0001 * l2_reg



def edge_aware_loss(pred, target, edge_ratio=3.0, kernel_size=3):
    """
    Args:
        pred: [B,1,H,W] 未激活的logits
        target: [B,H,W] 二值化掩码
    """
    # 维度处理
    if target.dim() == 4:
        target = target.squeeze(1)
    pred = pred.squeeze(1)  # [B,1,H,W] -> [B,H,W]
    target = target.float()

    # 转为4D [B,1,H,W]（适配conv2d）
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)

    # 边缘检测核
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                        device=target.device, dtype=torch.float32)
    pad = kernel_size // 2

    # 计算边缘
    with torch.no_grad():
        target_edges = F.conv2d(target, kernel, padding=pad)
        target_edges = (target_edges > 0) & (target_edges < kernel_size ** 2)

    # 加权BCEWithLogits
    edge_weight = torch.where(target_edges, edge_ratio, 1.0)
    return F.binary_cross_entropy_with_logits(
        pred, target, weight=edge_weight, reduction='mean'
    )


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets, hard_mask=None):
        # 维度处理
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        inputs = inputs.squeeze(1)
        targets = targets.float()

        # 焦点权重计算
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma

        # 应用hard_mask
        if hard_mask is not None:
            if hard_mask.dim() == 4:
                hard_mask = hard_mask.squeeze(1)
            focal_weight = focal_weight * hard_mask

        # 加权Dice
        probas = torch.sigmoid(inputs)
        intersection = (probas * targets * focal_weight).sum()
        denominator = (probas * focal_weight).sum() + (targets * focal_weight).sum()
        return 1 - (2. * intersection + self.smooth) / (denominator + self.smooth)


class SolarPanelLoss(nn.Module):
    def __init__(self, edge_ratio=3.0, gamma=2.0):
        super().__init__()
        self.dice_bce = DynamicDiceBCELoss()
        self.edge_ratio = edge_ratio
        self.gamma = gamma

    def forward(self, model, pred, target):
        """
        Args:
            pred: [B,1,H,W] 模型输出的logits
            target: [B,H,W] 二值化掩码
        """
        # 维度校验
        assert pred.dim() == 4, f"Pred should be [B,1,H,W], got {pred.shape}"
        assert target.dim() in [3, 4], f"Target should be [B,H,W] or [B,1,H,W], got {target.shape}"
        target = target.float()

        # 主损失
        main_loss = self.dice_bce(model, pred, target)

        # 边缘增强
        edge_loss = edge_aware_loss(pred, target, self.edge_ratio)

        # 难样本挖掘
        with torch.no_grad():
            probas = torch.sigmoid(pred.squeeze(1))  # [B,H,W]
            if target.dim() == 4:
                target = target.squeeze(1)
            hard_mask = (torch.abs(probas - target) > 0.5).float()

        # 焦点损失
        focal_loss = FocalDiceLoss(self.gamma)(pred, target, hard_mask)

        return main_loss + 0.3 * edge_loss + 0.2 * focal_loss

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super().__init__()
#         self.smooth = smooth
#
#     def forward(self, pred, target):
#         pred = pred.squeeze(1)  # [B,1,H,W] -> [B,H,W]
#         probas = torch.sigmoid(pred)
#         intersection = (probas * target).sum()
#         denominator = probas.sum() + target.sum()
#         return 1 - (2. * intersection + self.smooth) / (denominator + self.smooth)