"""
Test a sample of adversarial examples generated with an attack over a model.
"""

import os
import sys
import time
import math
import yaml
import random
import functools
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import torch
from torchvision import transforms
import timm
from argparse import ArgumentParser
from scipy.fftpack import dct, idct, rfft, irfft
import numpy as np

from robustbench.loaders import default_loader
from robustbench.data import CustomImageFolder, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from utils import reproducibility, setup_logger, read_yaml
from art.defences.preprocessor import JpegCompression, FeatureSqueezing

def argparser():
    
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tm",
        "--target-model",
        choices=[
            "vgg16.tv_in1k",
            "inception_v3.tv_in1k",
            "resnet50.tv2_in1k",
            "inception_v3.tf_adv_in1k",
            "inception_resnet_v2.tf_ens_adv_in1k",
            "random-padding",
            "jpeg",
            "bit-reduction",
            "neural-representation-purifier",
            "vgg19.tv_in1k",
            "resnet152.tv2_in1k",
            "mobilenetv2_140.ra_in1k",
        ],
        required=True,
        type=str,
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=30,
        help="10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory name (not path)",
        required=True,
        type=str,
    )
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("-g", "--gpu-id", type=int, default=0)
    parser.add_argument("-bs", "--batch-size", type=int, required=True)
    parser.add_argument("--test-samples", action="store_true")
    parser.add_argument("--indices-samples", default=None, type=str)
    parser.add_argument("--test-transferability", action="store_true")
    return parser


def load_adversarial_dataset(data_dir: str):
    """
    Load the images and its classes/label from a directory structure like the
    following:
        root/211/1.png
        root/211/2872.png
        root/211/4561.png
        root/567/1275.png
        root/567/21341.png
        root/567/4513.png
    so, the following directory name to the root folder is the label
    of the images it contains, and the images file name corresponds to 
    its index in the ImageNet dataset.
    """
    
    transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    classes_unique = [d.name for d in os.scandir(data_dir) if d.is_dir()]
    logger.info(f"Read {len(classes_unique)} classes")
    samples, classes, names = [], [], []

    for class_unique in classes_unique:
        class_dir = os.path.join(data_dir, class_unique)

        logger.debug(f"Reading directory {class_dir}")

        for image_path in os.scandir(class_dir):
            sample = default_loader(image_path.path)
            sample = transformations(sample)
            samples.append(sample.unsqueeze(0))
            classes.append(int(class_unique))
            names.append(int(image_path.name.split(".")[0]))

    samples = torch.vstack(samples)
    classes = torch.tensor(classes)
    logger.info(f"{len(samples)} samples have been read")
    return samples, classes, names


def load_imagenet(
        model_name: str,
        names,
        threat_model: str = "Linf",
):
    """
    Load the images from the ImageNet dataset whose index is in ``names``.
    """

    if model_name in ["random-padding"]:
        model_name = "inception_resnet_v2.tf_ens_adv_in1k"
    elif model_name in ["jpeg", "bit-reduction", "neural-representation-purifier"]:
        model_name = "mobilenetv2_140.ra_in1k"
    
    prepr = get_preprocessing(
        BenchmarkDataset("imagenet"), ThreatModel(threat_model), model_name, None
    )
    
    dataset = CustomImageFolder(
        "../data/imagenet/val",
        transform=prepr,
    )

    x_test, y_test = [], []

    for index in names:
        x, y, _ = dataset[index]
        x_test.append(x.unsqueeze(0))
        y_test.append(y)

    x_test = torch.vstack(x_test)
    y_test = torch.tensor(y_test)
    
    return x_test, y_test

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


class RandomResizingPaddingModel:
    def __init__(self, base_model):
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

class FeatureSqueezingModel:

    def __init__(self, base_model):
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

class JPEGCompressionModel:

    def __init__(self, base_model):
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


class NRPModel:

    def __init__(self, base_model):
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

def load_model(
        model_name: str,
):
    #transformations = None
    if timm.is_model(model_name):
        model = timm.create_model(model_name, pretrained=True)
        #data_config = resolve_model_data_config(model, use_test_size=True)
        #transformations = create_transform(**data_config, is_training=False)
    elif model_name == "random-padding":
        model = timm.create_model("inception_resnet_v2.tf_ens_adv_in1k", pretrained=True)
        model = RandomResizingPaddingModel(model)
    elif model_name == "jpeg":
        model = timm.create_model("mobilenetv2_140.ra_in1k", pretrained=True)
        model = JPEGCompressionModel(model)
    elif model_name == "bit-reduction":
        model = timm.create_model("mobilenetv2_140.ra_in1k", pretrained=True)
        model = FeatureSqueezingModel(model)
    elif model_name == "neural-representation-purifier":
        model = timm.create_model("inception_resnet_v2.tf_ens_adv_in1k", pretrained=True)
        model = NRPModel(model)
    else:
        raise ValueError(f"The value {model_name} is not a valid model name.")
    
    return model

def main(args):
    
    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available() and args.gpu_id is not None
        else torch.device("cpu")
    )
    batch_size = args.batch_size
    target_model = load_model(args.target_model)
    target_model = target_model.to(device)
    target_model.eval()
    adv_sample, classes, names = load_adversarial_dataset(args.data_dir)

    stime = time.time()

    random.seed()
    random_n = random.randint(0, 99999999)
    good_indices_filename = f"good_indices{random_n}.yaml"
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    good_indices_path = os.path.join(
        output_dir,
        good_indices_filename,
    )

    if args.test_samples:
        # Get those real images from ImageNet that have an adversary example
        real_sample, real_classes = load_imagenet(args.target_model, names)
        nexamples = len(adv_sample)
        acc = torch.ones((nexamples,), dtype=bool)
        target_sample_indices_all = torch.arange(nexamples, dtype=torch.long)
        nbatches = math.ceil(nexamples / batch_size)

        for idx in range(nbatches):
            logger.info(msg=f"idx = {idx}")
            begin = idx * batch_size
            end = min((idx + 1) * batch_size, nexamples)
            target_sample_indices = target_sample_indices_all[begin:end]
            logit = target_model(real_sample[target_sample_indices].clone().to(device)).cpu()
            preds = logit.argmax(1)
            acc[target_sample_indices] = preds == real_classes[target_sample_indices]

        clean_accuracy = acc.sum().item() / acc.shape[0] * 100
        logger.info(f"clean accuracy on real images from the adversarial dataset: {clean_accuracy:.2f}%")

        good_indices = [name for index, name in enumerate(names) if acc[index]]
        with open(good_indices_path, "w") as file:
            yaml.dump({"indices": good_indices}, file)

    else:
        good_indices = torch.tensor(read_yaml(good_indices_path).indices)
    
    if args.test_transferability:
        
        # Get those adversarial images whose original image is correcly
        # classified by the target model
        indices = [index for index, name in enumerate(names) if name in good_indices]
        logger.info(f"using {len(indices)} images from the total {len(names)}.")
        adv_sample = adv_sample[indices]
        classes = classes[indices]
        
        nexamples = len(adv_sample)
        acc = torch.ones((nexamples,), dtype=bool)
        target_sample_indices_all = torch.arange(nexamples, dtype=torch.long)
        nbatches = math.ceil(nexamples / batch_size)

        for idx in range(nbatches):
            logger.info(msg=f"idx = {idx}")
            begin = idx * batch_size
            end = min((idx + 1) * batch_size, nexamples)
            target_sample_indices = target_sample_indices_all[begin:end]
            logit = target_model(adv_sample[target_sample_indices].clone().to(device)).cpu()
            preds = logit.argmax(1)
            acc[target_sample_indices] = preds == classes[target_sample_indices]
            #inds = logit.argsort(1)

        logger.info(f"accuracy: {acc.sum().item() / acc.shape[0] * 100:.2f}%")

        accuracy = 100 * acc.sum().item() / acc.shape[0]
        attack_success_rate = 100 - accuracy
        msg = (
            f"adversarial images:{args.data_dir}\n"
            f"number of images used in target model: {len(indices)}\n"
            f"target model = {args.target_model}\n"
            f"total time (sec) = {time.time() - stime:.3f}\n"
            f"transferability ASR(%) = {attack_success_rate:.2f}\n"
            f"good indices file path: {good_indices_path}\n"
        )

        
        short_summary_path = os.path.join(output_dir, f"{args.target_model}.txt")
        with open(short_summary_path, "a") as f:
            f.write(msg)
            f.write("\n")

        logger.info(msg)


if __name__ == '__main__':

    sys.path.append("../src")
    reproducibility()
    parser = argparser()
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("The batch size must be greater than 1.")
    
    if not args.test_samples and not args.indices_samples:
        raise ValueError(
            "If we are not going to obtain the images from data_dir" \
            "that are correctly classified by the the target model, then you need to provide a file with" \
            "the indices (name of the images in the data_dir) with the --indices-samples option."
        )

    logger = setup_logger.setLevel(args.log_level)
    torch.set_num_threads(args.nthreads)
    os.environ["OMP_NUM_THREADS"] = str(args.nthreads)

    main(args)