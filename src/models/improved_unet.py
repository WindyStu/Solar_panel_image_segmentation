import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedDoubleConv(nn.Module):
    """改进的双卷积块，添加SE注意力机制"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # SE注意力机制
        self.se = SELayer(out_channels)
        
    def forward(self, x):
        x = self.double_conv(x)
        x = self.se(x)
        return x


class SELayer(nn.Module):
    """Squeeze-and-Excitation注意力机制"""
    
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedUNet(nn.Module):
    """改进的UNet，针对太阳能板分割优化"""
    
    def __init__(self, n_channels=3, n_classes=1, features=[32, 64, 128, 256]):
        super(ImprovedUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        in_channels = n_channels
        for feature in features:
            self.downs.append(ImprovedDoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ImprovedDoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    features[-1] * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ImprovedDoubleConv(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)
        
        # 添加深度监督
        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(features[0], n_classes, kernel_size=1),
            nn.Conv2d(features[1], n_classes, kernel_size=1),
            nn.Conv2d(features[2], n_classes, kernel_size=1)
        ])

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder with deep supervision
        deep_outputs = []
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
            
            # Deep supervision
            if idx // 2 < len(self.deep_supervision):
                deep_out = self.deep_supervision[idx // 2](x)
                deep_outputs.append(deep_out)

        # Final output
        final_output = self.final_conv(x)
        
        if self.training:
            return final_output, deep_outputs
        else:
            return final_output


class LightweightUNet(nn.Module):
    """轻量级UNet，参数量更少，适合简单任务"""
    
    def __init__(self, n_channels=3, n_classes=1):
        super(LightweightUNet, self).__init__()
        self.inc = ImprovedDoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ImprovedDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
            self.conv = ImprovedDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ImprovedDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    model = LightweightUNet()
    model.eval()
    x = torch.randn(1, 3, 160, 128)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {model(x).shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")