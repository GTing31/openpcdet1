from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1,)


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.ReLU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        if token_mixer == nn.Identity:
            self.token_mixer = nn.Identity()
        else:
            self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

def downsample_block(in_channels, out_channels, norm_fn, stride=2, padding=1,act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False),
        norm_fn(out_channels),
        act_layer(),
    )

class PillarInceptionNextBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        ResBlock = BasicBlock
        Block = downsample_block

        # 定义主干网络的各个卷积层，使用 MetaNeXtBlock
        self.conv1 = nn.Sequential(
            Block(64, 64, norm_fn=norm_fn, stride=2),
            # ResBlock(32, 32, norm_fn=norm_fn),
            # ResBlock(32, 32, norm_fn=norm_fn),
            MetaNeXtBlock(dim=64, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
            MetaNeXtBlock(dim=64, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
            MetaNeXtBlock(dim=64, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
        )

        self.conv2 = nn.Sequential(
            downsample_block(64, 128, norm_fn=norm_fn, stride=2),
            # ResBlock(64, 64, norm_fn=norm_fn),
            # ResBlock(64, 64, norm_fn=norm_fn),
            MetaNeXtBlock(dim=128, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
            # MetaNeXtBlock(dim=128, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
            MetaNeXtBlock(dim=128, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
            # MetaNeXtBlock(dim=64, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),

        )

        self.conv3 = nn.Sequential(
            downsample_block(128, 256, norm_fn=norm_fn),
            # ResBlock(128, 128, norm_fn=norm_fn),
            # ResBlock(128, 128, norm_fn=norm_fn),
            MetaNeXtBlock(dim=256, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
            MetaNeXtBlock(dim=256, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),

        )

        # self.conv4 = nn.Sequential(
        #     downsample_block(256, 256, norm_fn=norm_fn),
        #     # ResBlock(256, 256, norm_fn=norm_fn),
        #     # ResBlock(256, 256, norm_fn=norm_fn),
        #     MetaNeXtBlock(dim=256, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
        #     # MetaNeXtBlock(dim=256, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
        # )
        #
        # self.conv5 = nn.Sequential(
        #     downsample_block(256, 256, norm_fn=norm_fn),
        #     # ResBlock(256, 256, norm_fn=norm_fn),
        #     # ResBlock(256, 256, norm_fn=norm_fn),
        #     # MetaNeXtBlock(dim=256, token_mixer=InceptionDWConv2d, norm_layer=norm_fn),
        #
        # )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 64,
            'x_conv2': 128,
            'x_conv3': 256,
            # 'x_conv4': 256,
            # 'x_conv5': 256
        }

    def forward(self, batch_dict):
        # 获取输入特征，形状应为 [batch_size, channels, H, W]
        x = batch_dict['spatial_features']  # 假设输入已经是 2D 的特征图

        # 打印初始输入信息
        # print(f"Input x: shape={x.shape}, min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")

        # 经过初始卷积层，调整通道数为 32
        # print(f"After input_conv: shape={x.shape}")

        # 逐层经过主干网络
        x_conv1 = self.conv1(x)  # 输出通道数 32，stride 1
        # print(f"After conv1: shape={x_conv1.shape}")

        x_conv2 = self.conv2(x_conv1)  # 输出通道数 64，stride 2
        # print(f"After conv2: shape={x_conv2.shape}")

        x_conv3 = self.conv3(x_conv2)  # 输出通道数 128，stride 4
        # print(f"After conv3: shape={x_conv3.shape}")

        # x_conv4 = self.conv4(x_conv3)  # 输出通道数 256，stride 8
        # # x_conv4 = F.pad(x_conv4, (0, 1, 0, 1), mode='constant', value=0)# 为了保证特征图大小能被 2 整除
        # # print(f"After conv4: shape={x_conv4.shape}")
        #
        #
        #
        # x_conv5 = self.conv5(x_conv4)  # 输出通道数 256，stride 16
        # print(f"After conv5: shape={x_conv5.shape}")

        # 更新 batch_dict，包含多尺度特征
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                # 'x_conv4': x_conv4,
                # 'x_conv5': x_conv5,
            },
            'multi_scale_2d_strides': {
                'x_conv1': 2,
                'x_conv2': 4,
                'x_conv3': 8,
                # 'x_conv4': 8,
                # 'x_conv5': 16,
            }
        })

        return batch_dict

class InceptionResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, branch_ratio=0.25):
        super(InceptionResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 使用 InceptionDWConv2d 代替标准的 3x3 卷积
        self.inception_dwconv = InceptionDWConv2d(
            in_channels=out_channels,
            square_kernel_size=3,
            band_kernel_size=11,
            branch_ratio=branch_ratio
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)  # 1x1 卷积
        out = self.bn1(out)
        out = self.relu(out)

        out = self.inception_dwconv(out)  # InceptionDWConv2d 模块
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # 1x1 卷积
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


