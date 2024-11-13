"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[6, 4, 4, 2], mlp_ratio=4, drop_path_rate=0.0, **kwargs):
        super().__init__()

        self.in_channel = 32
        # stem layer
        # self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        # self.norm = nn.BatchNorm2d(self.in_channel)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # # self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            # 'x_conv5': 256
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, batch_dict):
        x = batch_dict['spatial_features']
        # print("x shape:", x.shape)  # 打印特徵張量形狀
        # x = self.stem(x)
        # for stage in self.stages:
        #     x = stage(x)
        #
        #
        # print(self.stages)
        x_conv1 = self.stages[0](x)

        # print("x_conv1 shape:", x_conv1.shape)  # 打印特徵張量形狀

        x_conv2 = self.stages[1](x_conv1)

        # print("x_conv2 shape:", x_conv2.shape)

        x_conv3 = self.stages[2](x_conv2)

        # print("x_conv3 shape:", x_conv3.shape)

        x_conv4 = self.stages[3](x_conv3)
        # print("x_conv4 shape:", x_conv4.shape)


        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                # 'x_conv5': x_conv5,
            },
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                # 'x_conv5': 16,
            }
        })
        return batch_dict
        # x = torch.flatten(self.avgpool(self.norm(x)), 1)
        # return self.head(x)


# @register_model
# def starnet_s1(pretrained=False, **kwargs):
#     model = StarNet(24, [2, 2, 8, 3], **kwargs)
#     if pretrained:
#         url = model_urls['starnet_s1']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["state_dict"])
#     return model
#
#
# @register_model
# def starnet_s2(pretrained=False, **kwargs):
#     model = StarNet(32, [1, 2, 6, 2], **kwargs)
#     if pretrained:
#         url = model_urls['starnet_s2']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["state_dict"])
#     return model
#
#
# @register_model
# def starnet_s3(pretrained=False, **kwargs):
#     model = StarNet(32, [2, 2, 8, 4], **kwargs)
#     if pretrained:
#         url = model_urls['starnet_s3']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["state_dict"])
#     return model
#
#
# @register_model
# def starnet_s4(pretrained=False, **kwargs):
#     model = StarNet(32, [3, 3, 12, 5], **kwargs)
#     if pretrained:
#         url = model_urls['starnet_s4']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["state_dict"])
#     return model
#
#
# # very small networks #
# @register_model
# def starnet_s050(pretrained=False, **kwargs):
#     return StarNet(16, [1, 1, 3, 1], 3, **kwargs)
#
#
# @register_model
# def starnet_s100(pretrained=False, **kwargs):
#     return StarNet(20, [1, 2, 4, 1], 4, **kwargs)
#
#
# @register_model
# def starnet_s150(pretrained=False, **kwargs):
#     return StarNet(24, [1, 2, 4, 2], 3, **kwargs)