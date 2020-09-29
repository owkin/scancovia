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
import SimpleITK as sitk
from skimage import measure
from lungmask import mask as lungmask


class NiftiPathToSitk():
    """
    Reads a NifTI file from the 'nii_path' key
    New key: image
    """

    def __call__(self, sample):
        sample['image'] = sitk.ReadImage(sample['nii_path'])
        return sample


class SitkResample(object):
    """
    Resamples sample['image'] to a new pixel spacing with a trilinear interpolation

    Parameters
    ----------
        output_spacing: tuple(float)
            new spacing in mm/voxel
    """

    def __init__(self, output_spacing):
        self.output_spacing = np.array(output_spacing).astype(float)

    def __call__(self, sample):
        image = sample['image']
        input_spacing = np.array(image.GetSpacing())
        input_size = np.array(image.GetSize())
        output_size = input_size * input_spacing / self.output_spacing

        image = sitk.Resample(
            image,
            output_size.astype(int).tolist(),
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            tuple(self.output_spacing),
            image.GetDirection(),
            0.,
            image.GetPixelID()
        )

        sample['image'] = image
        return sample


class CreateLungMask():
    """
    Creates a segmentation mask of the lungs using the U-Net R231 model
    Mask values are 0 for background, 1 for right lung and 2 for left lung
    New key: mask

    Parameters
    ----------
        device: str
            'cuda' to run on GPU, 'cpu' to run on CPU
    """

    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        mask = lungmask.apply(sample['image'], force_cpu=self.device == 'cpu')
        mask = mask.transpose((1, 2, 0))[::-1, :, :]
        sample['mask'] = mask
        return sample


class SitkToNumpy(object):
    """
    Converts sample['image'] from sitk image to numpy array
    """

    def __call__(self, sample):
        image = sample['image']
        sample['pixel_spacing'] = image.GetSpacing()
        image = sitk.GetArrayFromImage(image).transpose(1, 2, 0)[::-1, :, :]
        sample['image'] = image
        return sample


class CleanLungMask():
    """
    Remove spurious connected components from the lung mask:
        - only keeps the two 3D biggest components for left and right lung
        - deletes 2D components with 95% voxels greater than -100 HU are removed
    Finally converts sample['mask'] to a binary boolean mask

    Parameters
    ----------
        hu_threshold: int
        percentile: int
    """

    def __init__(self, hu_threshold=-100, percentile=5):
        self.threshold = hu_threshold
        self.percentile = percentile

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        clean_mask = np.zeros(mask.shape, dtype=bool)

        # For left and right lung, only keep the biggest connected component
        for i in [1, 2]:
            label_mask = measure.label(mask == i)
            regions = measure.regionprops(label_mask)
            regions = sorted(regions, key=lambda x: x.area, reverse=True)
            if len(regions):
                clean_mask += (label_mask == regions[0].label)

        # For each slice, only keep components with low enough HU units
        for i in range(mask.shape[2]):
            label_mask_i = measure.label(clean_mask[..., i])
            regions = measure.regionprops(label_mask_i)
            for r in regions:
                rr, cc = r.coords.T
                if np.percentile(image[rr, cc, i], self.percentile) > self.threshold:
                    clean_mask[rr, cc, i] = False

        sample['mask'] = clean_mask
        return sample


class ApplyLungMask():
    """
    Crops image to slices with lung
    Set voxels outside of lung to a constant value

    Parameters
    ----------
        constant_values: int
    """

    def __init__(self, constant_values):
        self.value = constant_values

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        image[mask == 0] = self.value
        slices_to_extract = mask.sum((0, 1)) > 0
        sample['image'] = image[..., slices_to_extract]
        sample['mask'] = mask[..., slices_to_extract]
        return sample


class CropPadAxial():
    """
    Crops or pads image and mask to (output_size, output_size, Z)

    Parameters
    ----------
        output_size: int
        constant_values: int
    """

    def __init__(self, output_size=512, constant_values=-1024):
        self.output_size = output_size
        self.value = constant_values

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        assert image.shape[0] == image.shape[1]
        input_size = image.shape[0]

        if input_size > self.output_size:
            # If the image it too big, crop
            center = input_size // 2
            width = self.output_size // 2
            sl = slice(center - width, center + width)
            image = image[sl, sl, :]
            mask = mask[sl, sl, :]

        elif input_size < self.output_size:
            # If the image is too small, pad
            diff = self.output_size - np.array(image.shape)
            diff[2] = 0
            pad_left = diff // 2
            pad_right = diff - pad_left
            pad = tuple(zip(pad_left, pad_right))

            image = np.pad(image, pad, constant_values=self.value)
            mask = np.pad(mask, pad, constant_values=False)

        sample['image'] = image
        sample['mask'] = mask
        return sample
