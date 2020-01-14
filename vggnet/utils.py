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


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def vggnet_params(model_name):
    """ Map VGGNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients: cfg, res, batch_norm
        "vggnet-b11":    ("A", 224, False),
        "vggnet-b13":    ("B", 224, False),
        "vggnet-b16":    ("D", 224, False),
        "vggnet-b19":    ("E", 224, False),
        "vggnet-b11_bn": ("A", 224, True),
        "vggnet-b13_bn": ("B", 224, True),
        "vggnet-b16_bn": ("D", 224, True),
        "vggnet-b19_bn": ("E", 224, True),
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
        c, s, b = vggnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        global_params = vggnet(arch=c, image_size=s, batch_norm=b)
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
    print(f"Loaded pretrained weights for {model_name}")

