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

from torch.utils import model_zoo


def adjust_learning_rate(optimizer, epoch, args):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.lr * (0.1 ** (epoch // 30))
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr


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


def vgg_params(model_name):
  """ Map vggNet model name to parameter vggnet. """
  params_dict = {
    #                  cfg, batch_norm
    "vgg11": ("A", False),
    "vgg13": ("B", False),
    "vgg16": ("D", False),
    "vgg19": ("E", False),
    "vgg11_bn": ("A", True),
    "vgg13_bn": ("B", True),
    "vgg16_bn": ("D", True),
    "vgg19_bn": ("E", True),
  }
  return params_dict[model_name]


def get_model_params(model_name):
  """ Get the block args and global params for a given model """
  if model_name.startswith("vgg"):
    cfg, batch_norm = vgg_params(model_name)
  else:
    raise NotImplementedError("model name is not pre-defined: %s" % model_name)
  return cfg, batch_norm


urls_map = {
  "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
  "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
  "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
  "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
  "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
  "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
  "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
  "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


def load_pretrained_weights(model, model_name):
  """ Loads pretrained weights, and downloads if loading for the first time. """
  state_dict = model_zoo.load_url(urls_map[model_name])
  model.load_state_dict(state_dict)
  print(f"Loaded pretrained weights for {model_name}.")


def get_parameter_number(model):
  total_num = sum(p.numel() for p in model.parameters())
  trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Total parameters: {total_num / 1000000:.1f}M")
  print(f"Trainable parameters: {trainable_num / 1000000:.1f}M")
