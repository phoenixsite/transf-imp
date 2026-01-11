
import functools
import random
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import timm

from art.defences.preprocessor import JpegCompression, FeatureSqueezing

def padding_layer_iyswim(inputs, shape):
    h_start = shape[0]
    w_start = shape[1]
    output_short = shape[2]
    input_shape = inputs.shape
    input_short = float(min(input_shape[2:3]))
    input_long = float(max(input_shape[2:3]))
    output_long = math.ceil(
        output_short * input_long / input_short)
    output_height = (input_shape[2] >= input_shape[3]) * output_long +\
        (input_shape[2] < input_shape[3]) * output_short
    output_width = (input_shape[2] >= input_shape[3]) * output_short +\
        (input_shape[2] < input_shape[3]) * output_long
    return torch.nn.functional.pad(inputs, [h_start, output_height - h_start - input_shape[2], w_start, output_width - w_start - input_shape[3]])


class RandomResizingPaddingModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.IMAGE_RESIZE = 331
        self.model = base_model
        resize_shape_ = random.randint(310, 331)
        self.shape_tensor = [random.randint(0, self.IMAGE_RESIZE - resize_shape_), random.randint(0, self.IMAGE_RESIZE - resize_shape_), self.IMAGE_RESIZE]
        self.filter = padding_layer_iyswim

    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, X):
        X_filtered = self.filter(X, self.shape_tensor)
        Y_pred = self.model(X_filtered)
        return Y_pred
    
    @property
    def default_cfg(self):
        return self.model.default_cfg

class FeatureSqueezingModel(nn.Module):

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.filter = FeatureSqueezing((0, 1))

    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, X):
        X_filtered, _ = self.filter(X.cpu())
        x_tensor = torch.tensor(X_filtered, device=X.device)
        Y_pred = self.model(x_tensor)
        return Y_pred
    
    @property
    def default_cfg(self):
        return self.model.default_cfg

class JPEGCompressionModel(nn.Module):

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.filter = JpegCompression((0, 1), apply_fit=False)

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def __call__(self, X):
        X_filtered, _ = self.filter(X.cpu())
        x_tensor = torch.tensor(X_filtered, device=X.device)
        Y_pred = self.model(x_tensor)
        return Y_pred
    
    @property
    def default_cfg(self):
        return self.model.default_cfg


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class NRP(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(NRP, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))

        return trunk
    
class NRP_resG(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23):
        super(NRP_resG, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = make_layer(basic_block, nb)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.conv_last(self.recon_trunk(fea))
        return out


class NRPModel(nn.Module):

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.purifier = NRP_resG(3, 3, 64, 23)
        self.purifier.load_state_dict(torch.load('pretrained_purifiers/NRP_resG.pth'))
        self.filter = self.purify

    def purify(self, x):
        return self.purifier(x).detach()

    def eval(self):
        self.purifier.eval()
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)
        self.purifier = self.purifier.to(device)
        return self
    
    def __call__(self, X):
        X_filtered = self.filter(X)
        Y_pred = self.model(X_filtered)
        return Y_pred
    
    @property
    def default_cfg(self):
        return self.model.default_cfg

ADV_MODELS = {
    "random-padding": ("inception_resnet_v2.tf_ens_adv_in1k", RandomResizingPaddingModel),
    "jpeg": ("mobilenetv2_140.ra_in1k", JPEGCompressionModel),
    "bit-reduction": ("mobilenetv2_140.ra_in1k", FeatureSqueezingModel),
    "neural-representation-purifier": ("inception_resnet_v2.tf_ens_adv_in1k", NRPModel),
}

def get_model(model_name):
    """
    Load a model.

    Currently, only the models available in the package timm
    are supported.
    """

    if timm.is_model(model_name):
        model = timm.create_model(model_name, True)
    elif model_name in ADV_MODELS:
        adv_build = ADV_MODELS[model_name]
        model = timm.create_model(adv_build[0], True)
        model = adv_build[1](model)
    else:
        raise ValueError(f"The '{model_name}' is not supported.")
    
    return model