import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class AddNoise(nn.Module):
    """模拟低分辨率图像噪声"""

    def forward(self, x):
        if self.training:
            x += torch.randn_like(x) * 0.02
        return x


class _ConvBnReLU(nn.Sequential):
    """卷积+BN+ReLU三件套"""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False)
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module("relu", nn.ReLU(inplace=True))


class _ImagePool(nn.Module):
    """全局平均池化分支"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        h = self.pool(x)
        h = self.conv(h)
        return F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)


class _ASPP(nn.Module):
    """改进的轻量ASPP模块"""

    def __init__(self, in_ch, out_ch, rates=[1, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([
            _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1),  # 1x1卷积
            _ConvBnReLU(in_ch, out_ch, 3, 1, rates[0], rates[0]),  # rate1
            _ConvBnReLU(in_ch, out_ch, 3, 1, rates[1], rates[1]),  # rate3
            _ImagePool(in_ch, out_ch)  # 图像池化
        ])

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages], dim=1)


class _Bottleneck(nn.Module):
    """轻量瓶颈结构"""

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super().__init__()
        mid_ch = out_ch // 2  # 压缩中间通道数
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, relu=False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, relu=False)
            if downsample else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h, inplace=True)


class _ResLayer(nn.Sequential):
    """残差层（支持多网格）"""

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super().__init__()
        multi_grids = multi_grids or [1] * n_layers

        for i in range(n_layers):
            self.add_module(
                f"block{i + 1}",
                _Bottleneck(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    stride if i == 0 else 1,
                    dilation * multi_grids[i],
                    downsample=(i == 0)
                )
            )


class DeepLabV3Plus(nn.Module):
    """优化后的DeepLabV3+（适配160x128输入）"""

    def __init__(self, n_classes=1, output_stride=8):
        super().__init__()
        assert output_stride == 8, "只支持output_stride=8的低分辨率模式"

        # 通道配置（原版1/2）
        ch = [32, 64, 128, 256]  # stem, layer2, layer3, layer4

        # 下采样策略（s=stride, d=dilation）
        s = [1, 2, 1, 1]  # layer3不下采样
        d = [1, 1, 2, 4]  # 通过膨胀卷积扩大感受野

        # 主干网络
        self.stem = nn.Sequential(
            AddNoise(),
            _ConvBnReLU(3, ch[0], 3, 1, 1, 1),  # 修改初始卷积
            nn.MaxPool2d(2, 2, 0, ceil_mode=True)
        )
        self.layer1 = _ResLayer(2, ch[0], ch[1], s[0], d[0])  # 2 blocks
        self.layer2 = _ResLayer(2, ch[1], ch[2], s[1], d[1])  # 3 blocks
        self.layer3 = _ResLayer(2, ch[2], ch[3], s[2], d[2])  # 3 blocks

        # ASPP
        self.aspp = _ASPP(ch[3], 64, rates=[1, 3, 6])
        self.aspp_proj = _ConvBnReLU(64 * 4, 128, 1, 1, 0, 1)

        # 解码器（多层特征融合）
        self.low_level_proj = _ConvBnReLU(ch[1], 32, 1, 1, 0, 1)
        self.decoder = nn.Sequential(
            _ConvBnReLU(128 + 32, 128, 3, 1, 1, 1),
            _ConvBnReLU(128, 128, 3, 1, 1, 1),
            nn.Conv2d(128, n_classes, 1)
        )

        # 亚像素上采样（可选）
        self.upsample = nn.PixelShuffle(2) if n_classes % 4 == 0 else None

    def forward(self, x):
        # 编码器
        h = self.stem(x)  # /2 → 80x64
        h1 = self.layer1(h)  # /2 → 40x32 (skip)
        h2 = self.layer2(h1)  # /1 → 40x32
        h3 = self.layer3(h2)  # /1 → 40x32

        # ASPP
        h_aspp = self.aspp(h3)
        h_aspp = self.aspp_proj(h_aspp)  # 128ch

        # 解码器
        h_low = self.low_level_proj(h1)  # 32ch
        h_aspp = F.interpolate(
            h_aspp, size=h_low.shape[2:], mode='bilinear', align_corners=False
        )
        h_out = torch.cat([h_aspp, h_low], dim=1)
        h_out = self.decoder(h_out)

        # 上采样到原图
        h_out = F.interpolate(
            h_out, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        if self.upsample:
            h_out = self.upsample(h_out)

        return h_out


if __name__ == "__main__":
    model = DeepLabV3Plus(n_classes=1)
    model.eval()
    x = torch.randn(1, 3, 160, 128)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {model(x).shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")