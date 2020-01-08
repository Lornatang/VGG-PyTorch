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

import torch
import torch.nn as nn
from .utils import get_model_params
from .utils import load_pretrained_weights


class VGGNet(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGGNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def from_name(cls, model_name, num_classes=1000, init_weights=True):
      cls._check_model_name_is_valid(model_name)
      cfg, batch_norm = get_model_params(model_name)
      model = _vgg(cfg, batch_norm, num_classes, init_weights)
      return model

    @classmethod
    def from_pretrained(cls, model_name, init_weights=False):
      model = cls.from_name(model_name, 1000, init_weights)
      load_pretrained_weights(model, model_name)
      return model

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
      """ Validates model name. None that pretrained weights are only available for
      the first four models (vgg{i} for i in 11,13,16,19) at the moment. """
      valid_models = ['vgg'+str(i) for i in ["11", "11_bn", 
                                             "13", "13_bn", 
                                             "16", "16_bn", 
                                             "19", "19_bn"]]
      if model_name not in valid_models:
          raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
    
    @classmethod
    def load_weights(cls, model_name, model_path, num_classes, init_weights=True, **kwargs):
      model = cls.from_name(model_name, num_classes, init_weights)
      checkpoint = torch.load(model_path)
      model.load_state_dict(checkpoint['state_dict'])
      return model


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg, batch_norm, num_classes, init_weights, **kwargs):
    model = VGGNet(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes, init_weights, **kwargs)
    return model
