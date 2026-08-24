from typing import Any, Callable, List, Optional, Type, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=2,
        stride=stride,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class DeBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = (
            deconv2x2(width, width, stride, groups, dilation)
            if stride == 2
            else conv3x3(width, width, stride, groups, dilation)
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.upsample is not None:
            identity = self.upsample(x)

        return self.relu(out + identity)


class EfficientChannelAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid_channels = max(in_channels // reduction, out_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.gelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return self.gelu(out + residual)


class LightweightMLP(nn.Module):
    def __init__(
        self,
        in_channels: int = 2048,
        out_channels_list: Optional[List[int]] = None,
        target_sizes: Optional[List[tuple[int, int]]] = None,
        reduction: int = 2,
    ) -> None:
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [1024, 512]
        if target_sizes is None:
            target_sizes = [(16, 16), (32, 32)]

        mid_channels = in_channels // reduction
        self.target_sizes = target_sizes
        self.shared_transform = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.branch1 = nn.Conv2d(mid_channels, out_channels_list[0], kernel_size=1, bias=False)
        self.branch2 = nn.Conv2d(mid_channels, out_channels_list[1], kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        shared_feat = self.shared_transform(x)
        out1 = self.branch1(shared_feat)
        out2 = self.branch2(shared_feat)

        out1 = F.interpolate(out1, size=self.target_sizes[0], mode="bilinear", align_corners=False)
        out2 = F.interpolate(out2, size=self.target_sizes[1], mode="bilinear", align_corners=False)
        return out1, out2


class PDDDecoder(nn.Module):
    def __init__(
        self,
        block: Type[Union[DeBottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        channel_reduction: int = 4,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, "
                f"got {replace_stride_with_dilation}"
            )

        self._norm_layer = norm_layer
        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            64,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )

        self.up_dim_v_r1 = EfficientChannelAdapter(192, 256, reduction=channel_reduction)
        self.up_dim_v_r2 = EfficientChannelAdapter(384, 512, reduction=channel_reduction)
        self.up_dim_v_r3 = EfficientChannelAdapter(768, 1024, reduction=channel_reduction)
        self.up_dim_v_r4 = EfficientChannelAdapter(768, 2048, reduction=channel_reduction)
        self.up_2factor = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mlp = LightweightMLP(
            in_channels=2048,
            out_channels_list=[1024, 512],
            target_sizes=[(16, 16), (32, 32)],
            reduction=2,
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, DeBottleneck):
                    nn.init.constant_(module.bn3.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[DeBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                upsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, y, z, res: int, feature_list: bool = False):
        if res != 10:
            raise NotImplementedError(f"Only res=10 is supported by PDDDecoder, got res={res}")

        vs_fa = self.up_dim_v_r4(z[3])
        fuss_0 = vs_fa + x
        prior_f_1, prior_f_2 = self.mlp(fuss_0)

        vs_fb = self.up_2factor(self.up_dim_v_r3(z[2]))
        fuss_a = vs_fb + y[2]
        vs_fc = self.up_2factor(self.up_dim_v_r2(z[1]))
        fuss_b = vs_fc + y[1]
        vs_fd = self.up_2factor(self.up_dim_v_r1(z[0]))
        fuss_c = vs_fd + y[0]

        vssm_fa = self.layer1(fuss_0)
        vssm_fb = self.layer2(vssm_fa + prior_f_1)
        vssm_fc = self.layer3(vssm_fb + prior_f_2)

        res_fa = self.layer1(fuss_0)
        res_fb = self.layer2(res_fa)
        res_fc = self.layer3(res_fb)

        if feature_list:
            return [
                vssm_fc,
                vssm_fb,
                vssm_fa,
                fuss_c,
                fuss_b,
                fuss_a,
                res_fc,
                res_fb,
                res_fa,
            ]

        return vssm_fc


def pdd_decoder(
    pretrained: bool = False,
    progress: bool = True,
    channel_reduction: int = 4,
    **kwargs: Any,
) -> PDDDecoder:
    del pretrained, progress
    kwargs["width_per_group"] = 64 * 2
    kwargs["channel_reduction"] = channel_reduction
    return PDDDecoder(DeBottleneck, [2, 2, 2, 2], **kwargs)
