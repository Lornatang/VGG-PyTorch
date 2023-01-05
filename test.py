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
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import model
import test_config
from dataset import CUDAPrefetcher, ImageDataset
from utils import load_pretrained_state_dict, accuracy, Summary, AverageMeter, ProgressMeter


def build_model(
        model_arch_name: str = test_config.model_arch_name,
        num_classes: int = test_config.model_num_classes,
        device: torch.device = torch.device("cpu"),
) -> nn.Module:
    vgg_model = model.__dict__[model_arch_name](num_classes=num_classes)
    vgg_model = vgg_model.to(device)

    return vgg_model


def load_dataset(
        test_image_dir: str = test_config.test_image_dir,
        resized_image_size=test_config.resized_image_size,
        crop_image_size=test_config.crop_image_size,
        dataset_mean_normalize=test_config.dataset_mean_normalize,
        dataset_std_normalize=test_config.dataset_std_normalize,
        device: torch.device = torch.device("cpu"),
) -> CUDAPrefetcher:
    test_dataset = ImageDataset(test_image_dir,
                                resized_image_size,
                                crop_image_size,
                                dataset_mean_normalize,
                                dataset_std_normalize,
                                "Test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=test_config.batch_size,
                                 shuffle=False,
                                 num_workers=test_config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return test_prefetcher


def test(
        model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        device: torch.device,
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"Test: ")

    # Put the exponential moving average model in the verification mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device, non_blocking=True)
            target = batch_data["target"].to(device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = model(images)

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0].item(), batch_size)
            acc5.update(top5[0].item(), batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % test_config.test_print_frequency == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    return acc1.avg


def main() -> None:
    device = torch.device(test_config.device)

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Initialize the model
    vgg_model = build_model(device=device)
    vgg_model, _, _, _, _, _ = load_pretrained_state_dict(vgg_model, test_config.model_weights_path)

    # Start the verification mode of the model.
    vgg_model.eval()

    test(vgg_model, test_prefetcher, device)


if __name__ == "__main__":
    main()
