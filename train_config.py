# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
seed = 0
device = "cuda:0"
# Model configure
model_arch_name = "vgg19"
model_num_classes = 1000

# Experiment name, easy to save weights and log files
exp_name = f"VGG19-ImageNet_1K"

# Dataset address
train_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_train"
valid_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"

dataset_mean_normalize = (0.485, 0.456, 0.406)
dataset_std_normalize = (0.229, 0.224, 0.225)

resized_image_size = 256
crop_image_size = 224
batch_size = 128
num_workers = 4

# The address to load the pretrained model
pretrained_model_weights_path = "./results/pretrained_models/VGG19-ImageNet_1K-53d9a5c0.pth.tar"

# Incremental training and migration training
resume_model_weights_path = ""

# Total num epochs
epochs = 600

# Loss parameters
loss_label_smoothing = 0.1

# Optimizer parameter
model_lr = 0.1
model_momentum = 0.9
model_weight_decay = 2e-05
model_ema_decay = 0.99998

# Learning rate scheduler parameter
lr_scheduler_T_0 = epochs // 4
lr_scheduler_T_mult = 1
lr_scheduler_eta_min = 5e-5

# How many iterations to print the training/validate result
train_print_frequency = 10
valid_print_frequency = 10
