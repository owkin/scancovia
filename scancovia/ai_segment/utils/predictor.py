"""
ScanCovIA
Copyright (C) 2020 INRIA, OWKIN

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details
"""

import torch
import torch.nn.functional as F
import numpy as np


def run_models(scan, unet_lung, resnet_seg, unet_seg, device, batch_size=32):
    """
    Runs inference of our models over a scan

    Arguments:
    ---------
    scan: numpy.ndarray
        The scan of interest
    unet_lung: torch.nn.Module
        U-Net model for left/right lungs segmentation
    resnet_seg: torch.nn.Module
        ResNet model for lesion segmentation
    unet_seg: torch.nn.Module
        U-Net model for lesion segmentation
    device: str
        'cuda' for to run on GPU, 'cpu' to run on VPU
    batch_size: int
    """

    # Preprocess
    scan += 1024
    scan[scan < 0] = 0
    scan /= 1000
    scan = scan.transpose(2, 0, 1)  # shape (Z, H, W)
    scan = torch.tensor(scan, device=device)

    with torch.no_grad():
        # Submodel 1: U-Net for lung segmentation
        output = []
        for batch in torch.split(scan, batch_size // 2, dim=0):
            pred = unet_lung(batch.unsqueeze(1) - 1)  # input shape (B, 1, H, W)
            pred = (pred > 0).sum(1)  # fuse left and right lung
            pred = pred.data.cpu().numpy()
            output.append(pred)
        lungs_mask = np.concatenate(output)

        # Submodel 2: ResNet for lesion segmentation
        output = []
        for batch in torch.split(scan, batch_size, dim=0):
            pred = resnet_seg(batch.unsqueeze(0))  # input shape (1, B, H, W)
            pred = F.interpolate(pred, size=[batch.shape[0], 512, 512], mode='trilinear')[0]
            pred = F.softmax(pred, dim=0)
            pred = pred.data.cpu().numpy()
            output.append(pred)
        resnet_output = np.concatenate(output, axis=1)

        # Submodel 3: U-Net for lesion segmentation
        output = []
        for z in range(len(scan) - 2):
            batch = scan[z: z + 3].unsqueeze(0).transpose(-1, -2)  # input shape (1, 3, W, H)
            pred = unet_seg(batch)[0]
            pred = F.softmax(pred, dim=0)
            pred = pred.transpose(-1, -2).data.cpu().numpy()
            output.append(pred)
        unet_output = np.concatenate(output, axis=1)
        unet_output = np.pad(unet_output, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='constant')

        # Combine submodels
        output = (unet_output + resnet_output) / 2
        output = output.argmax(0)
        output[lungs_mask == 0] = 0

        output = output.transpose((1, 2, 0))
        lungs_mask = lungs_mask.transpose((1, 2, 0))
        del scan

    return lungs_mask, output


def compute_disease_extent(segmentation_mask):
    """
    Returns the disease extent for each class.

    Arguments:
    ---------
    segmentation_mask: np.ndarray
        The segmented lung

    Returns:
    --------
    sample: dict
        A dictionnary containing the percentage of the extent of each class.
    """

    vol_lungs = float(np.count_nonzero(segmentation_mask > 0))

    label_dict = {
        1: 'GGO AiSegment',
        2: 'Consolidation AiSegment',
        3: 'Crazy paving AiSegment',
        4: 'Sane lung AiSegment',
    }

    sample = dict((lesion_class, 0.) for lesion_class in label_dict.values())

    for label_index, label_counts in zip(*np.unique(segmentation_mask, return_counts=True)):
        if label_index in label_dict:
            lesion_class = label_dict[label_index]
            sample[lesion_class] = label_counts / vol_lungs
    sample['Disease extent AiSegment'] = 1 - sample['Sane lung AiSegment']

    return sample
