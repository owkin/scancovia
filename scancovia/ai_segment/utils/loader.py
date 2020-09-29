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

import numpy as np
import nibabel as nib


def nib_raw_loader(nii_path):
    """
    Loads a nifti file as numpy

    Arguments
    ---------
    nii_path: str
        Path to the nifti file to load

    Returns
    -------
    scan: numpy.ndarray
        The loaded scan (volume)
    """
    scan_r = nib.load(nii_path)
    scan = scan_r.get_fdata()

    for axis in range(3):
        if scan_r.affine[axis, axis] < 0:
            scan = np.flip(scan, axis=axis)
    scan = scan.transpose((1, 0, 2))[::-1, ::-1, :]
    scan = scan.astype(np.float32)

    return scan
