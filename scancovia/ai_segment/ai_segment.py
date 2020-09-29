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

import numpy as np
import torch

from scancovia.ai_segment.utils.predictor import run_models, compute_disease_extent
from scancovia.ai_segment.utils.loader import nib_raw_loader
from scancovia.ai_segment.utils.get_models import get_unet_models, get_resnet50


class AiSegment():
    """
    Class for the AiSegment model

    Parameters
    ----------
    device: str
        'cuda' to run on GPU, 'cpu' to run on cpu

    Examples
    --------
    >>> from scancovia import AiSegment
    >>> ai_segment = AiSegment()
    >>> output = ai_segment('patient_exam.nii')
    """

    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Load models
        self.unet_seg, self.unet_lung = get_unet_models(device)
        self.resnet_seg = get_resnet50(device)


    def __call__(self, nii_path):
        """

        Parameters
        ----------
        nii_path: str
            Path to a NIfTI file. We recommend dcm2niix for DICOM folder conversion

        Returns
        -------
        sample: dict
            dictionnary with lesion volumes ratios and segmentation/lungs masks

        """

        scan = nib_raw_loader(nii_path)
        lungs_mask, segmentation_mask = run_models(
            np.copy(scan), self.unet_lung, self.resnet_seg, self.unet_seg, self.device)

        sample = compute_disease_extent(segmentation_mask)
        sample['segmentation_mask'] = segmentation_mask
        sample['lungs_mask'] = lungs_mask
        sample['image'] = scan

        return sample
