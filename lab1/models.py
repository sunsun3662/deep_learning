import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """老师原始示例风格的 CIFAR10 卷积网络（LeNet-like）。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 与老师 notebook 保持一致：两层卷积 + 两层池化 + 三层全连接。
        # 输入尺寸为 3x32x32，经两次 5x5 卷积和 2x2 池化后，特征图为 16x5x5。
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 卷积提取局部特征，ReLU 引入非线性，池化进行下采样。
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平后进入全连接分类头。
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    """简化版 ResNet 基本残差块。"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 主分支：3x3 -> BN -> ReLU -> 3x3 -> BN。
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径分支：当尺寸或通道变化时，用 1x1 卷积对齐。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接核心：F(x) + x，能缓解深层网络梯度退化问题。
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNetCustom(nn.Module):
    """个人实现的 ResNet-18 风格网络（每个 stage 两个 BasicBlock）。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # CIFAR10 输入分辨率较小，stem 使用 3x3 卷积且不做初始最大池化。
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 四个 stage，空间尺寸按 32->16->8->4 递减，通道逐步翻倍。
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        # 第一个 block 负责可能的下采样与通道变换，后续 block 尺寸不变。
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class DenseLayer(nn.Module):
    """DenseNet 中的单层：输出会与输入在通道维拼接。"""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        # DenseNet 常见 bottleneck 设计：1x1 降维后接 3x3 卷积。
        inter_channels = growth_rate * 4
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """由多个 DenseLayer 串联组成，每层都接收前面所有层的特征。"""

    def __init__(self, in_channels: int, num_layers: int, growth_rate: int):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Transition(nn.Module):
    """DenseNet 过渡层：压缩通道并下采样，控制模型复杂度。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trans(x)


class DenseNetCustom(nn.Module):
    """个人实现的轻量 DenseNet。"""

    def __init__(self, num_classes: int = 10, growth_rate: int = 16):
        super().__init__()
        init_channels = 32
        self.stem = nn.Conv2d(3, init_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.db1 = DenseBlock(init_channels, num_layers=4, growth_rate=growth_rate)
        ch1 = self.db1.out_channels
        self.tr1 = Transition(ch1, ch1 // 2)

        self.db2 = DenseBlock(ch1 // 2, num_layers=4, growth_rate=growth_rate)
        ch2 = self.db2.out_channels
        self.tr2 = Transition(ch2, ch2 // 2)

        self.db3 = DenseBlock(ch2 // 2, num_layers=4, growth_rate=growth_rate)
        ch3 = self.db3.out_channels

        self.bn = nn.BatchNorm2d(ch3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.db1(x)
        x = self.tr1(x)
        x = self.db2(x)
        x = self.tr2(x)
        x = self.db3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class DepthwiseSeparableConv(nn.Module):
    """MobileNet 核心模块：深度卷积 + 逐点卷积。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # 深度卷积按通道独立提取空间特征，逐点卷积进行通道融合。
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetCustom(nn.Module):
    """个人实现的轻量 MobileNet 风格网络。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 通过 stride=2 的模块逐步下采样，最后全局池化接分类头。
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
            DepthwiseSeparableConv(512, 512, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Res2Block(nn.Module):
    """Res2Net 风格残差块：在单个残差块内部进行多尺度分支特征融合。"""

    def __init__(self, channels: int, scale: int = 4):
        super().__init__()
        if channels % scale != 0:
            raise ValueError("channels must be divisible by scale for Res2Block")

        self.scale = scale
        self.width = channels // scale
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True),
                )
                for _ in range(scale - 1)
            ]
        )

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.pre(x)
        splits = torch.split(out, self.width, dim=1)

        outputs = [splits[0]]
        for i in range(1, self.scale):
            if i == 1:
                y = self.convs[i - 1](splits[i])
            else:
                y = self.convs[i - 1](splits[i] + outputs[i - 1])
            outputs.append(y)

        out = torch.cat(outputs, dim=1)
        out = self.post(out)
        out = out + identity
        return self.relu(out)


class Res2NetCustom(nn.Module):
    """个人实现的轻量 Res2Net：stage 内采用 Res2Block 提升多尺度表示。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            Res2Block(64, scale=4),
            Res2Block(64, scale=4),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.stage2 = nn.Sequential(
            Res2Block(128, scale=4),
            Res2Block(128, scale=4),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.stage3 = nn.Sequential(
            Res2Block(256, scale=4),
            Res2Block(256, scale=4),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    # 统一模型工厂：训练脚本只依赖模型名，便于切换与批量实验。
    name = name.lower()
    if name == "cnn":
        return CNN(num_classes=num_classes)

    if name == "resnet":
        return ResNetCustom(num_classes=num_classes)

    if name == "densenet":
        return DenseNetCustom(num_classes=num_classes)

    if name == "mobilenet":
        return MobileNetCustom(num_classes=num_classes)

    if name == "res2net":
        return Res2NetCustom(num_classes=num_classes)

    supported = "cnn, resnet, densenet, mobilenet, res2net"
    raise ValueError(f"Unsupported model: {name}. Supported: {supported}")
