import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math


# TSM Module Definition
class TemporalShiftModule(nn.Module):
    def __init__(self, channels, shift_div=8, shift_place='channel'):
        super(TemporalShiftModule, self).__init__()
        self.channels = channels
        self.shift_div = shift_div
        self.shift_place = shift_place

        assert shift_place in ['block', 'channel'], "shift_place should be either 'block' or 'channel'"

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()

        # Dividing channels into multiple groups
        shift_size = channels // self.shift_div

        # If shifting is done in a block-wise manner
        if self.shift_place == 'block':
            # Shift the feature map across depth dimension
            x = x.view(batch_size, self.shift_div, shift_size, depth, height, width)
            x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
            x = x.view(batch_size, channels, depth, height, width)

        # If shifting is done in a channel-wise manner
        elif self.shift_place == 'channel':
            # Shift across the depth (temporal shift)
            x = x.view(batch_size, self.shift_div, shift_size, depth, height, width)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.view(batch_size, channels, depth, height, width)

        return x


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, depth, height, width)
    # permute
    x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, shift_div=8, shift_place='channel'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2
        self.tsm = TemporalShiftModule(oup_inc, shift_div, shift_place)  # TSM module

        if self.stride == 1:
            self.banch2 = nn.Sequential(
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        # Apply Temporal Shift Module (TSM) on the output after the pointwise convolution
        out = self.tsm(out)

        return channel_shuffle(out, 2)

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=600, sample_size=112, width_mult=2., pretrain_path=None):
        super(ShuffleNetV2, self).__init__()
        assert sample_size % 16 == 0

        self.stage_repeats = [4, 8, 4]
        if width_mult == 0.25:
            self.stage_out_channels = [-1, 24, 32, 64, 128, 1024]
        elif width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError("Unsupported width_mult value.")

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, stride=(1, 2, 2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1x1_bn(input_channel, self.stage_out_channels[-1])
        self.avgpool = nn.AvgPool3d((2, 1, 1), stride=1)

        # Adding TSM at the end of the network
        self.tsm = TemporalShiftModule(self.stage_out_channels[-1], shift_div=8, shift_place='channel')  # TSM module

        self.pretrain_path = pretrain_path

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.conv_last(out)

        # Apply TSM at the last layer
        out = self.tsm(out)  # Temporal Shift at the end

        if out.size(2) == 2:
            out = self.avgpool(out)

        return out

    def load_pretrain(self):
        print("backbone3D : shufflenetv2 pretrained loading skipped (disabled by code_TSM)", flush=True)
        return


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def build_shufflenetv2(config):
    width_mult = config['BACKBONE3D']['SHUFFLENETv2']['width_mult']
    assert width_mult in [0.25, 0.5, 1.0, 1.5, 2.0], "wrong width_mult of shufflenetv2!"
    pretrain_dict = config['BACKBONE3D']['SHUFFLENETv2']['PRETRAIN']

    if width_mult == 0.25:
        pretrain_path = pretrain_dict['width_mult_0.25x']
    elif width_mult == 0.5:
        pretrain_path = pretrain_dict['width_mult_0.5x']
    elif width_mult == 1.0:
        pretrain_path = pretrain_dict['width_mult_1.0x']
    elif width_mult == 1.5:
        pretrain_path = pretrain_dict['width_mult_1.5x']
    elif width_mult == 2.0:
        pretrain_path = pretrain_dict['width_mult_2.0x']

    return ShuffleNetV2(width_mult=width_mult, pretrain_path=pretrain_path)
