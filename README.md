# VGGNet

### Update (January 15, 2020)

This update allows you to use NVIDIA's Apex tool for accelerated training. By default choice `hybrid training precision` + `dynamic loss amplified` version, if you need to learn more and details about `apex` tools, please visit https://github.com/NVIDIA/apex.

### Update (January 9, 2020)

This update adds a visual interface for testing, which is developed by pyqt5. At present, it has realized basic functions, and other functions will be gradually improved in the future.

### Update (January 6, 2020)

This update adds a modular neural network, making it more flexible in use. It can be deployed to many common dataset classification tasks. Of course, it can also be used in your products.

### Overview
This repository contains an op-for-op PyTorch reimplementation of [VGGNet](https://arxiv.org/pdf/1409.1556.pdf).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained VGGNet models 
 * Use VGGNet models for classification or feature extraction 

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an VGGNet on your own dataset
 * Export VGGNet models for production
 
### Table of contents
1. [About VGGNet](#about-vgg)
2. [Installation](#installation)
3. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
4. [Contributing](#contributing) 

### About VGGNet

If you're new to VGGNets, here is an explanation straight from the official PyTorch implementation: 

In this work we investigate the effect of the convolutional network depth on its
accuracy in the large-scale image recognition setting. Our main contribution is
a thorough evaluation of networks of increasing depth using an architecture with
very small (3 × 3) convolution filters, which shows that a significant improvement
on the prior-art configurations can be achieved by pushing the depth to 16–19
weight layers. These findings were the basis of our ImageNet Challenge 2014
submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations
generalise well to other datasets, where they achieve state-of-the-art results. We
have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

### Installation

Install from source:
```bash
git clone https://github.com/lornatang/VGGNet
cd VGGNet
python setup.py install
``` 

### Usage

#### Loading pretrained models

Load an vgg11 network:
```python
from vgg import VGGNet
model = VGGNet.from_name("vgg11")
```

Load a pretrained vgg11: 
```python
from vgg import VGGNet
model = VGGNet.from_pretrained("vgg11")
```

Details about the models are below (for CIFAR10 dataset): 

|      *Name*       |*# Params*|*Top-1 Acc.*|*Pretrained?*|
|:-----------------:|:--------:|:----------:|:-----------:|
|     `vgg11`       |  132.9M  |    91.1    |      √      |
|     `vgg13`       |   133M   |    92.8    |      √      |
|     `vgg16`       |  138.4M  |    92.6    |      √      |
|     `vgg19`       |  143.7M  |    92.3    |      √      |
|-------------------|----------|------------|-------------|
|     `vgg11_bn`    |  132.9M  |    92.2    |      √      |
|     `vgg13_bn`    |   133M   |    94.2    |      √      |
|     `vgg16_bn`    |  138.4M  |    93.9    |      √      |
|     `vgg19_bn`    |  143.7M  |    93.7    |      √      |


#### Example: Classification

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`. 

```python
import json
import urllib

import torch
import torchvision.transforms as transforms
from PIL import Image

from vggnet import VGGNet

input_image = Image.open("img.jpg")

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

labels_map = json.load(open("labels_map.txt"))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify with VGGNet
model = VGGNet.from_pretrained("vgg11")
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    logits = model(input_batch)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print("-----")
for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print(f"{label:<75} ({prob * 100:.2f}%)")
```

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 