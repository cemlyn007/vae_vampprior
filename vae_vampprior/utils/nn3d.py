import torch.nn as nn


# =======================================================================================================================
class GatedConv3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv3d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


# =======================================================================================================================
class Conv3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None,
                 bias=True):
        super(Conv3d, self).__init__()

        self.activation = activation
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, x):
        h = self.conv(x)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out


# =======================================================================================================================
class Conv3dBN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(Conv3dBN, self).__init__()

        self.activation = activation
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm3d(output_channels)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out


# =======================================================================================================================
class ResUnitBN(nn.Module):
    '''
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep Residual Networks", https://arxiv.org/abs/1603.05027
    The unit used here is called the full pre-activation.
    '''

    def __init__(self, number_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ReLU(True)):
        super(ResUnitBN, self).__init__()

        self.activation = activation
        self.bn1 = nn.BatchNorm3d(number_channels)

        self.conv1 = nn.Conv3d(number_channels, number_channels, kernel_size, stride, padding, dilation)

        self.bn2 = nn.BatchNorm3d(number_channels)
        self.conv2 = nn.Conv3d(number_channels, number_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        residual = x

        h_bn_1 = self.bn1(x)
        h_act_bn_1 = self.activation(h_bn_1)
        h_1 = self.conv1(h_act_bn_1)

        h_bn_2 = self.bn2(h_1)
        h_act_bn_2 = self.activation(h_bn_2)
        h_2 = self.conv2(h_act_bn_2)

        out = h_2 + residual

        return out


# =======================================================================================================================
class ResizeConv3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, scale_factor=2,
                 activation=None):
        super(ResizeConv3d, self).__init__()

        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        h = self.upsamplingNN(x)
        h = self.conv(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out


# =======================================================================================================================
class ResizeConv3dBN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, scale_factor=2,
                 activation=None):
        super(ResizeConv3dBN, self).__init__()

        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm3d(output_channels)

    def forward(self, x):
        h = self.upsamplingNN(x)
        h = self.conv(h)
        h = self.bn(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out


# =======================================================================================================================
class ResizeGatedConv3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, scale_factor=2,
                 activation=None):
        super(ResizeGatedConv3d, self).__init__()

        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = GatedConv3d(input_channels, output_channels, kernel_size, stride, padding, dilation,
                                activation=activation)

    def forward(self, x):
        h = self.upsamplingNN(x)
        out = self.conv(h)

        return out


# =======================================================================================================================
class GatedConvTranspose3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, output_padding=0, dilation=1,
                 activation=None):
        super(GatedConvTranspose3d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose3d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)
        self.g = nn.ConvTranspose3d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


# =======================================================================================================================
class GatedResUnit(nn.Module):
    def __init__(self, input_channels, activation=None):
        super(GatedResUnit, self).__init__()

        self.activation = activation
        self.conv1 = GatedConv3d(input_channels, input_channels, 3, 1, 1, 1, activation=activation)
        self.conv2 = GatedConv3d(input_channels, input_channels, 3, 1, 1, 1, activation=activation)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)

        return h2 + x


# =======================================================================================================================
class MaskedConv3d(nn.Conv3d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)


# =======================================================================================================================
class MaskedGatedConv3d(nn.Module):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedGatedConv3d, self).__init__()

        self.h = MaskedConv3d(mask_type, *args, **kwargs)
        self.g = MaskedConv3d(mask_type, *args, **kwargs)

    def forward(self, x):
        h = self.h(x)
        g = nn.Sigmoid()(self.g(x))
        return h * g


# =======================================================================================================================
class MaskedResUnit(nn.Module):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedResUnit, self).__init__()

        self.act = nn.ReLU(True)

        self.h1 = MaskedConv3d(mask_type, *args, **kwargs)
        self.h2 = MaskedConv3d(mask_type, *args, **kwargs)

        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(64)

    def forward(self, x):
        h1 = self.bn1(x)
        h1 = self.act(h1)
        h1 = self.h1(h1)

        h2 = self.bn2(h1)
        h2 = self.act(h2)
        h2 = self.h2(h2)
        return x + h2
