import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath


# -----------------------------
# Utilities
# -----------------------------
def _init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class LayerNorm2d(nn.Module):
    """
    LayerNorm over channel维 (等价于对 NHWC 的 LayerNorm, 但在 NCHW 上实现——避免 permute).
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


# -----------------------------
# Stem & Downsample (全 NCHW，无 permute)
# -----------------------------
class StemLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, act_layer=nn.GELU,
                 norm_layer=partial(LayerNorm2d, eps=1e-6)):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, 2, 1, bias=False)
        self.norm1 = norm_layer(mid)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(mid, out_channels, 3, 2, 1, bias=False)
        self.norm2 = norm_layer(out_channels)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x  # NCHW


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 norm_layer=partial(LayerNorm2d, eps=1e-6)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)
        self.norm = norm_layer(out_channels)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x  # NCHW


# -----------------------------
# GatedCNNBlock (Linear -> 1x1Conv; 去 permute)
# -----------------------------
class GatedCNNBlock(nn.Module):
    """
    等价于你原本的门控 MLP + DWConv，但完全在 NCHW 上：
      - fc1/fc2 用 1x1 Conv 实现（对每个像素的线性变换）
      - 规范化用 LayerNorm2d（跨通道）
      - 保留 DropPath
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(LayerNorm2d, eps=1e-6), act_layer=nn.GELU,
                 drop_path=0.):
        super().__init__()
        hidden = int(expansion_ratio * dim)
        conv_channels = int(conv_ratio * dim)
        if conv_channels > hidden:
            raise ValueError(
                f"conv_ratio*dim ({conv_channels}) must be <= expansion_ratio*dim ({hidden})"
            )

        self.norm = norm_layer(dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=True)
        # split along C
        self.split_channels = (hidden, hidden - conv_channels, conv_channels)
        self.dwconv = nn.Conv2d(conv_channels, conv_channels, kernel_size,
                                padding=kernel_size // 2, groups=conv_channels, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(_init_weights)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)  # [B, 2*hidden, H, W]
        g, i, c = torch.split(x, self.split_channels, dim=1)  # 按通道切
        c = self.dwconv(c)
        x = torch.cat((i, c), dim=1)
        x = self.fc2(self.act(g) * x)
        return shortcut + self.drop_path(x)


# -----------------------------
# Backbone: MambaOut (保持接口，但全 NCHW)
# -----------------------------
class MambaOut(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=(3, 3, 9, 3), dims=(96, 192, 384, 576),
                 norm_layer=partial(LayerNorm2d, eps=1e-6),
                 act_layer=nn.GELU,
                 conv_ratio=1.0, kernel_size=7, drop_path_rate=0.,
                 output_norm=partial(LayerNorm2d, eps=1e-6),
                 out_indices=(1, 2, 3)):
        """
        out_indices: 返回哪些stage的特征 (默认后三个，与原逻辑一致)。
        """
        super().__init__()
        assert len(depths) == len(dims)
        self.num_stage = len(depths)
        self.out_indices = set(out_indices)

        down_dims = (in_chans,) + tuple(dims)
        downsample_layers = [StemLayer] + [DownsampleLayer] * (self.num_stage - 1)
        self.downsample_layers = nn.ModuleList([
            downsample_layers[i](down_dims[i], down_dims[i + 1],
                                 norm_layer=norm_layer, act_layer=act_layer) if i == 0
            else downsample_layers[i](down_dims[i], down_dims[i + 1], norm_layer=norm_layer)
            for i in range(self.num_stage)
        ])

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(*[
                GatedCNNBlock(dims[i], norm_layer=norm_layer, act_layer=act_layer,
                              kernel_size=kernel_size, conv_ratio=conv_ratio,
                              drop_path=dp_rates[cur + j]) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        # 可选：分类或导出时会用到
        self.out_norm = output_norm(dims[-1])

    @torch.jit.ignore
    def no_weight_decay(self):
        # 给 optimizer 提示（如 LayerNorm2d 的仿射参数）
        return set()

    def extract_features(self, x):
        """
        返回 list[p3, p4, p5] (N, C, H, W)
        """
        feats = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                feats.append(x)
        return feats  # NCHW 顺序

    def forward(self, x):
        # 与原先保持一致：外部模块（FPN/Head）使用 extract_features
        return self.extract_features(x)


# -----------------------------
# DepthwiseSeparableConv with Norm & Act
# -----------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 norm=True, act=True, act_layer=nn.SiLU):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = act_layer() if act else nn.Identity()
        self.apply(_init_weights)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# -----------------------------
# DarkFPN (对齐上采样尺寸，避免 shape 漂移)
# -----------------------------
class DarkFPN(nn.Module):
    def __init__(self, width):
        """
        约定：
          p3: width[2], p4: width[3], p5: width[4]
        width 形如: [3, C1, C2, C3, C4, C4]
        """
        super().__init__()
        c2, c3, c4 = width[2], width[3], width[4]

        self.h1 = DepthwiseSeparableConv(c4 + c3, c3, 3, 1)  # up(p5)+p4 -> c3
        self.h2 = DepthwiseSeparableConv(c3 + c2, c2, 3, 1)  # up(h1)+p3 -> c2

        self.h3 = DepthwiseSeparableConv(c2, c2, 3, 2)       # down(h2) -> c2
        self.h4 = DepthwiseSeparableConv(c2 + c3, c3, 3, 1)  # + h1 -> c3

        self.h5 = DepthwiseSeparableConv(c3, c3, 3, 2)       # down(h4) -> c3
        self.h6 = DepthwiseSeparableConv(c3 + c4, c4, 3, 1)  # + p5 -> c4

    def forward(self, x_tuple):
        p3, p4, p5 = x_tuple  # NCHW

        # 上采样到精确 spatial size，避免由 stride 导致的奇偶数漂移
        up_p5 = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        h1 = self.h1(torch.cat([up_p5, p4], dim=1))

        up_h1 = F.interpolate(h1, size=p3.shape[-2:], mode='nearest')
        h2 = self.h2(torch.cat([up_h1, p3], dim=1))

        h4 = self.h4(torch.cat([self.h3(h2), h1], dim=1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], dim=1))
        return h2, h4, h6  # 对应 (P3,P4,P5) 的增强特征


# -----------------------------
# YOLO Neck wrapper
# -----------------------------
class YOLO(nn.Module):
    def __init__(self, width, depth, pretrain_path=None):
        super().__init__()
        dims = width[1:5]  # [C1, C2, C3, C4]
        self.backbone = MambaOut(dims=dims, depths=depth, out_indices=(1, 2, 3))
        self.fpn = DarkFPN(width)
        self.pretrain_path = pretrain_path

    def forward(self, x):
        p3, p4, p5 = self.backbone.extract_features(x)
        return self.fpn((p3, p4, p5))

    def fuse(self):
        # 这里主要是 LN，不易与 Conv 融合；保持占位以兼容外部流水线
        return self

    def load_pretrain(self):
        # 如需加载外部权重，可在这里补充映射逻辑
        print("Pretrained backbone loading skipped.", flush=True)
        return


# -----------------------------
# Builder
# -----------------------------
def build_yolov8(config):
    ver = config['BACKBONE2D']['YOLOv8']['ver']
    assert ver in ['mamba-n', 'mamba-s', 'mamba-m', 'mamba-l', 'mamba-x'], "错误的YOLOv8版本！"

    pretrain_path = config['BACKBONE2D']['YOLOv8']['PRETRAIN']

    version_configs = {
        'mamba-n': ((1, 2, 2, 1), (3, 48, 96, 192, 384, 384)),
        'mamba-s': ((2, 2, 4, 2), (3, 64, 128, 256, 384, 384)),
        'mamba-m': ((3, 3, 9, 3), (3, 96, 192, 384, 576, 576)),
        'mamba-l': ((4, 4, 12, 4), (3, 128, 256, 512, 768, 768)),
        'mamba-x': ((6, 6, 18, 6), (3, 160, 320, 640, 960, 960)),
    }

    depth, width = version_configs.get(ver, ((3, 3, 9, 3), (3, 96, 192, 384, 576, 576)))
    return YOLO(width, depth, pretrain_path)
