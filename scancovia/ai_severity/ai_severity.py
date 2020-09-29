"""
ScanCovIA
Copyright (C) 2020 OWKIN

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
from torchvision import transforms
from pathlib import Path

from scancovia.ai_severity.preprocessing_transforms import NiftiPathToSitk, SitkResample, CreateLungMask, SitkToNumpy, \
    CleanLungMask, ApplyLungMask, CropPadAxial
from scancovia.ai_severity.models_transforms import EfficientNetExtractor, MoCoExtractor, \
    ApplyPCA, ApplyLogisticRegression, BlendPredictions


class AiSeverity():
    """
    Class for the AiSeverity model

    Parameters
    ----------
    device: str
        'cuda' to run on GPU, 'cpu' to run on cpu

    Examples
    --------
    >>> from scancovia import AiSeverity
    >>> ai_severity = AiSeverity(device='cuda')
    >>> risk = ai_severity('patient_exam.nii')
    """

    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        weights_path = Path(__file__).resolve(
        ).parents[2] / 'assets/ai_severity.npz'
        weights = np.load(weights_path)  # load trained parameters

        self.transforms = transforms.Compose([
            # Preprocessing transforms
            NiftiPathToSitk(),
            SitkResample(output_spacing=(0.703125, 0.703125, 10)),
            CreateLungMask(device),
            SitkToNumpy(),
            CleanLungMask(hu_threshold=-100, percentile=5),
            ApplyLungMask(constant_values=-1024),
            CropPadAxial(output_size=512, constant_values=-1024),
            # Models transforms
            EfficientNetExtractor(key='efficientnet',
                                  vmin=-1024, vmax=600, device=device),
            ApplyLogisticRegression(key='efficientnet', coef=weights['coef_efficientnet'],
                                    intercept=weights['intercept_efficientnet']),
            MoCoExtractor(key='moco', windows=[
                (-1000, 0), (0, 1000), (-1000, 4000)], device=device),
            ApplyPCA(
                key='moco', mean=weights['mean_PCA'], components=weights['components_PCA']),
            ApplyLogisticRegression(
                key='moco', coef=weights['coef_moco'], intercept=weights['intercept_moco']),
            BlendPredictions()
        ])

    def __call__(self, nii_path):
        """

        Parameters
        ----------
        nii_path: str
            Path to a NIfTI file. We recommend dcm2niix for DICOM folder conversion


        Returns
        -------
        float
            Risk prediction of the AiSeverity model

        """
        sample = {'nii_path': nii_path}
        sample = self.transforms(sample)
        # For more details, try to return the sample dictionnary !
        return sample['prediction']['blend']
