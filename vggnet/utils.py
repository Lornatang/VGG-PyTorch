# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import collections
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'cfg', 'image_size', 'batch_norm', 'dropout_rate', 'num_classes', 'init_weights'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def vggnet_params(model_name):
    """ Map VGGNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients: cfg, res, batch_norm, dropout_rate
        "vggnet-b11":    ("A", 224, False, 0.2),
        "vggnet-b13":    ("B", 224, False, 0.2),
        "vggnet-b16":    ("D", 224, False, 0.3),
        "vggnet-b19":    ("E", 224, False, 0.3),
        "vggnet-b11_bn": ("A", 224, True, 0.2),
        "vggnet-b13_bn": ("B", 224, True, 0.2),
        "vggnet-b16_bn": ("D", 224, True, 0.3),
        "vggnet-b19_bn": ("E", 224, True, 0.3),
    }
    return params_dict[model_name]


def vggnet(arch=None, image_size=None, batch_norm=None, dropout_rate=0.2, num_classes=1000, init_weights=False):
    """ Creates a vggnet model. """

    global_params = GlobalParams(
        cfg=arch,
        image_size=image_size,
        batch_norm=batch_norm,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        init_weights=init_weights
    )

    return global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('vggnet'):
        c, s, b, p = vggnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        global_params = vggnet(arch=c, image_size=s, batch_norm=b, dropout_rate=p)
    else:
        raise NotImplementedError(f"model name is not pre-defined: {model_name}.")
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return global_params


urls_map = {
    "vggnet-b11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vggnet-b13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vggnet-b16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vggnet-b19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vggnet-b11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vggnet-b13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vggnet-b16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vggnet-b19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(urls_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('classifier.6.weight')
        state_dict.pop('classifier.6.bias')
        model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights for {model_name}.")


def load_custom_weights(model, model_name):
    """ Loads custom weights, and train if loading for the first time. """
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded custom weights for {model_name}.")


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_num / 1000000:.1f}M")
    print(f"Trainable parameters: {trainable_num / 1000000:.1f}M")


def print_state_dict(model):
    print("----------------------------------------------------------")
    print("|                    state dict pram                     |")
    print("----------------------------------------------------------")
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    print("----------------------------------------------------------")


#########################################################################
############## HELPERS FUNCTIONS FOR TRAINING MODEL PARAMS ##############
#########################################################################


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
