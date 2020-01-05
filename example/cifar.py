# Copyright 2019 Lorna Authors. All Rights Reserved.
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

import argparse
import os
import random

import torch
import torchvision
import torchsummary
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.dataloader
import torchvision.datasets as dset
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch AlexNet Classifier')
parser.add_argument('--dataroot', type=str,
                    default="../datasets/cifar", help="dataset path.")
parser.add_argument('--name', type=str, default="cifar-10",
                    help="Dataset name. Default: cifar-10.")
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int,
                    default=128, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=32,
                    help='the height / width of the inputs image to network')
parser.add_argument('--num_classes', type=int, default=10,
                    help="number of dataset category.")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="learning rate.")
parser.add_argument('--epochs', type=int, default=1000, help="Train loop")
parser.add_argument('--phase', type=str, default='eval',
                    help="train or eval? default:`eval`")
parser.add_argument('--checkpoints_dir', default='../checkpoints',
                    help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()

try:
  os.makedirs(opt.checkpoints_dir)
except OSError:
  pass

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

# setup gpu driver
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model path
MODEL_PATH = os.path.join(opt.checkpoints_dir, f"{opt.name}.pth")

data_name = str(opt.name)

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


def adjust_learning_rate(initial_lr=None, optimizer=None, epoch=None, every_epoch=2.4, reduction_rate=0.97):
  """Sets the learning rate to the initial LR decayed by 0.97 every 2.4 epochs"""
  lr = initial_lr * (reduction_rate ** (epoch // every_epoch))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
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
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg19():
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg19', 'E', False)


def vgg19_bn():
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg19', 'E', True)


def train():
  try:
    os.makedirs(opt.checkpoints_dir)
  except OSError:
    pass

  if opt.name == "cifar-10":
    train_dataset = dset.CIFAR10(root=opt.dataroot,
                               download=True,
                               train=True,
                               transform=transforms.Compose([
                                 transforms.Resize(opt.img_size, interpolation=3),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                               ]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                 pin_memory=torch.cuda.is_available(),
                                                 shuffle=True, num_workers=int(opt.workers))
    opt.num_classes = 10
  elif opt.name == "cifar-100":
    train_dataset = dset.CIFAR100(root=opt.dataroot,
                                download=True,
                                train=True,
                                transform=transforms.Compose([
                                  transforms.Resize(opt.img_size, interpolation=3),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                 pin_memory=torch.cuda.is_available(),
                                                 shuffle=True, num_workers=int(opt.workers))
    opt.num_classes = 100
  else:
    print(parser.print_help())
    exit(0)

  # Load model
  if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DataParallel(vgg19_bn())
  else:
    model = vgg19_bn()

  # change num_classes for corresponding dataset
  model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, opt.num_classes),
  )

  # TODO: Load ImageNet weight
  if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
  model.to(device)
  torchsummary.summary(model, (3, 32, 32))

  ################################################
  # Set loss function and Adam optimizer
  ################################################
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

  for epoch in range(opt.epochs):
    adjust_learning_rate(opt.lr, optimizer, epoch)
    # train for one epoch
    print(f"\nBegin Training Epoch {epoch}")
    # Calculate and return the top-k accuracy of the model
    # so that we can track the learning process.
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, data in enumerate(train_dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, targets = data
      inputs = inputs.to(device)
      targets = targets.to(device)

      # compute output
      output = model(inputs)
      loss = criterion(output, targets)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output, targets, topk=(1, 5))
      losses.update(loss.item(), inputs.size(0))
      top1.update(prec1, inputs.size(0))
      top5.update(prec5, inputs.size(0))

      # compute gradients in a backward pass
      optimizer.zero_grad()
      loss.backward()

      # Call step of optimizer to update model params
      optimizer.step()
      
      if i % 50 == 0:
        print(f"Epoch [{epoch}] [{i}/{len(train_dataloader)}]\t"
              f"Loss {loss.item():.4f}\t"
              f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
              f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})")

    # save model file
    torch.save(model.state_dict(), MODEL_PATH)


def test():
  if opt.name == "cifar-10":
    test_dataset = dset.CIFAR10(root=opt.dataroot,
                               download=True,
                               train=False,
                               transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                               ]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                 pin_memory=torch.cuda.is_available(),
                                                 num_workers=int(opt.workers))
    opt.num_classes = 10
  elif opt.name == "cifar-100":
    test_dataset = dset.CIFAR100(root=opt.dataroot,
                                download=True,
                                train=False,
                                transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, 
                                                 pin_memory=torch.cuda.is_available(),
                                                 num_workers=int(opt.workers))
    opt.num_classes = 100
  else:
    print(parser.print_help())
    exit(0)

  # Load model
  if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DataParallel(vgg19_bn())
  else:
    model = vgg19_bn()

  # change num_classes for corresponding dataset
  model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, opt.num_classes),
  )

  # TODO: Load ImageNet weight
  if not os.path.exists(MODEL_PATH):
    print("Please train....")
    exit(0)
    
  model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
  model.to(device)
  print(model)

  # init value
  correct1 = 0.
  correct5 = 0.
  total = len(test_dataloader.dataset)
  with torch.no_grad():
    for i, data in enumerate(test_dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, targets = data
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)

      # cal top 1 accuracy
      prec1 = outputs.argmax(dim=1)
      correct1 += torch.eq(prec1, targets).sum().item()

      # cal top 5 accuracy
      maxk = max((1, 5))
      targets_resize = targets.view(-1, 1)
      _, prec5 = outputs.topk(maxk, 1, True, True)
      correct5 += torch.eq(prec5, targets_resize).sum().item()

  return correct1 / total, correct5 / total


if __name__ == '__main__':
  if opt.name is None:
    print(parser.print_help())
    exit(0)
  elif opt.phase == "train":
    train()
  elif opt.phase == "eval":
    print("Loading model successful!")
    Top1, Top5 = test()
    print(
      f"Top 1 accuracy of the network on the test images: {100 * Top1:.2f}%.\n"
      f"Top 5 accuracy of the network on the test images: {100 * Top5:.2f}%.\n")
