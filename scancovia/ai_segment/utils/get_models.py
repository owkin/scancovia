"""
ScanCovIA
Copyright (C) 2020 INRIA

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
from pathlib import Path
from collections import OrderedDict

from scancovia.ai_segment.models.unet_model import UNet
from scancovia.ai_segment.models.resnet import resnet50


def get_resnet50(device):
    """
    Gets the resnet50 model

    Returns
    -------
    model_lesions: torch.nn.Module
        model for the lesion segmentation
    """

    resnet_lesion = resnet50(sample_input_D=32, sample_input_H=32,
                   sample_input_W=32, num_seg_classes=5)

    # TODO: move to github release instead of git LFS ########
    path = Path(__file__).resolve().parents[4] / 'resnet_lesion.pth'
    ckpt = torch.load(path, map_location=device)
    new_ckpt = OrderedDict((k[len('module.'):], v) for k, v in ckpt.items())
    resnet_lesion.load_state_dict(new_ckpt)
    ##########################################################

    resnet_lesion.to(device)
    resnet_lesion.eval()
    return resnet_lesion


def get_unet_models(device):
    """
    Gets the unet models

    Returns
    -------
    model_lesions: torch.nn.Module
        model for the lesion segmentation
    model_lungs: torch.nn.Module
        model for the left/right lung segmentation
    """

    # Load lesion model
    unet_lesion = UNet(n_channels=3, n_classes=5, kdim=3)

    # TODO: move to torch hub. Currently in Github release
    path = Path(__file__).resolve().parents[4] / 'unet_lesion.pth'
    ckpt = torch.load(path, map_location=device)
    new_ckpt = OrderedDict((k[len('module.'):], v) for k, v in ckpt.items())
    unet_lesion.load_state_dict(new_ckpt)
    ##########################################################

    unet_lesion.to(device)
    unet_lesion.eval()

    # Load lung model
    unet_lung = UNet(n_channels=1, n_classes=2, volume=False, kdim=1)

    # TODO: move to torch hub. Currently in Github release
    path = Path(__file__).resolve().parents[4] / 'unet_lung.pth'
    ckpt = torch.load(path, map_location=device)
    new_ckpt = OrderedDict((k[len('module.'):], v) for k, v in ckpt.items())
    unet_lung.load_state_dict(new_ckpt)
    ##########################################################

    unet_lung.to(device)
    unet_lung.eval()

    return unet_lesion, unet_lung
