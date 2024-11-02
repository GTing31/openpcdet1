import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv3']
        x_conv4 = self.conv_layer(x_conv4)

        x_conv5 = spatial_features['x_conv5']
        # print('x_conv4:', x_conv4.shape)
        # print('x_conv5:', x_conv5.shape)

        # print('blocks:', self.deblocks[0])

        ups = [self.deblocks[0](x_conv4)]
        # print('ups:', ups[0].shape)

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        # print('x:', x.shape)
        x = self.blocks[0](x)
        # print('x:', x.shape)

        data_dict['spatial_features_2d'] = x


        return data_dict


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print('out:', out.shape)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out


class BaseBEVResBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

def post_act_block(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m
class ConcatResBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        Block = post_act_block

        # 初始通道数
        self.initial_channels = 64
        self.pillar_context_channels = 64  # 根据您的 `pillar_context` 通道数设置

        # 定义初始卷积层
        # self.conv_input = nn.Sequential(
        #     nn.Conv2d(input_channels, self.initial_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.initial_channels, eps=1e-3, momentum=0.01),
        #     nn.ReLU(inplace=True)
        # )
        #(64,496,432)
        # 第一层卷积块
        self.layer1 = nn.Sequential(
            BasicBlock(self.initial_channels, self.initial_channels, stride=1),
            BasicBlock(self.initial_channels, self.initial_channels, stride=1),
            BasicBlock(self.initial_channels, self.initial_channels, stride=1)#(64,496,432)
        )

        # 第一层的 1x1 卷积
        # self.conv1x1_1 = nn.Sequential(
        #     nn.Conv2d(self.initial_channels + self.pillar_context_channels, 128, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     nn.ReLU(inplace=True)
        # )
        self.layer2 = nn.Sequential(
            Block(64, 64, stride=2, norm_fn=norm_fn, kernel_size=3, padding=1),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(64 + self.pillar_context_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        # 第二层卷积块
        self.layer3 = nn.Sequential(
            Block(128, 128, stride=2, norm_fn=norm_fn, kernel_size=3, padding=1),
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 128, stride=1)
        )

        # 第二层的 1x1 卷积
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(128 + self.pillar_context_channels, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        # 第三层卷积块
        self.layer4 = nn.Sequential(
            Block(192, 192, stride=2, norm_fn=norm_fn, kernel_size=3, padding=1),
            BasicBlock(192, 192, stride=1),
            BasicBlock(192, 192, stride=1)
        )

        # 第三层的 1x1 卷积
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(192 + self.pillar_context_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        # 第四层卷积块
        self.layer5 = nn.Sequential(
            BasicBlock(256, 256, stride=1),
            BasicBlock(256, 256, stride=1),
            BasicBlock(256, 256, stride=1)
        )

        # 最终的特征通道数
        self.num_bev_features = 256

    def _concat_with_context(self, x, context):
        # 检查并调整尺寸
        if x.shape[2:] != context.shape[2:]:
            context = F.interpolate(context, size=x.shape[2:], mode='bilinear', align_corners=False)
        # 拼接
        x = torch.cat([x, context], dim=1)
        # print('after concat:', x.shape)
        return x

    def forward(self, data_dict):
        x = data_dict['spatial_features']  # [B, C, H, W]
        pillar_context = data_dict['pillar_context']  # List of [B, C_pillar, H_i, W_i]
        # print('x:', x.shape)
        # 第一层
        # x = self.conv_input(x)
        layer1_out = self.layer1(x)
        # print('after layer1:', layer1_out.shape)


        # 第二层
        layer2_out = self.layer2(layer1_out)
        layer2_out = self._concat_with_context(layer2_out, pillar_context[0])
        layer2_out = self.conv1x1_1(layer2_out)
        # print('after layer2:', layer2_out.shape)

        layer3_out = self.layer3(layer2_out)
        layer3_out = self._concat_with_context(layer3_out, pillar_context[1])
        layer3_out = self.conv1x1_2(layer3_out)
        # print('after layer3:', layer3_out.shape)

        layer4_out = self.layer4(layer3_out)
        layer4_out = self._concat_with_context(layer4_out, pillar_context[2])
        layer4_out = self.conv1x1_3(layer4_out)
        # print('after layer4:', layer4_out.shape)

        layer5_out = self.layer5(layer4_out)
        # 第三层

        x = layer5_out
        # print('after layer5:', x.shape)
        # print("concat_res_backbone finished")

        data_dict['spatial_features_2d'] = x
        return data_dict



class BaseBEVBackboneV1_SingleScale(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS  # [5]
        num_filters = self.model_cfg.NUM_FILTERS  # [256]
        assert len(layer_nums) == len(num_filters) == 1  # 只使用一个尺度

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

        # 添加上采样层
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=num_filters[0], out_channels=num_filters[0],
                kernel_size=2, stride=2, bias=False
            ),
            nn.BatchNorm2d(num_filters[0], eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        # 输出的通道数为256
        self.num_bev_features = num_filters[0]

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                multi_scale_2d_features
        Returns:
            data_dict
        """
        spatial_features = data_dict['multi_scale_2d_features']

        # 选取单尺度特征（例如 x_conv5）
        x = spatial_features['x_conv5']  # 根据需要选择合适的尺度

        # 经过对应的 block
        x = self.blocks[0](x)

        # 进行上采样
        x = self.upsample(x)
        # print('After upsampling:', x.shape)

        # 将结果更新到 data_dict 中
        data_dict['spatial_features_2d'] = x
        # print('spatial_features_2d:', x.shape)

        return data_dict