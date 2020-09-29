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

from collections import OrderedDict
import numpy as np
import torch
from torchvision.models.resnet import resnet50
from efficientnet_pytorch import EfficientNet


class EfficientNetExtractor():
    """
    EfficientNet feature extractor
    New key: sample['feature'][key] of shape (Z, 1280, H//32, W//32)

    Parameters
    ----------
        key: str
            name of the model
        vmin: int
            lower bound to clip HU units
        vmax: int
            upper bound to clip HU units
        device: str
            'cuda' to run on GPU, 'cpu' to run on CPU
    """

    def __init__(self, key, vmin, vmax, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.vmin = vmin
        self.vmax = vmax
        self.key = key

        # Load model
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model.eval()
        model.to(device)
        self.model = model

    def __call__(self, sample):
        image = sample['image']  # shape (H, W, Z)
        # Grayscale to RGB
        image = image[None, ...].repeat(3, 0).transpose(
            (3, 0, 1, 2))  # shape (Z, C, H, W)
        # Clip HU and [0, 1] normalization
        image = np.clip(image, self.vmin, self.vmax)
        image = (image - self.vmin) / (self.vmax - self.vmin)
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean.reshape(1, 3, 1, 1)) / \
            std.reshape(((1, 3, 1, 1)))
        # Convert to Tensor
        image = torch.from_numpy(image).type(torch.FloatTensor).to(self.device)

        # Run extractor
        with torch.no_grad():
            features = self.model.extract_features(image).detach(
            ).cpu().numpy()  # shape (Z, 1280, H//32, W//32)

        sample.setdefault('features', {})[self.key] = features
        return sample


class MoCoExtractor():
    """
    MoCoExtractor feature extractor (resnet50)
    New key: sample['feature'][key] of shape (Z, 2048, H//32, W//32)

    Parameters
    ----------
        key: str
            name of the model
        windows: list of tuple of int
            Lower and upper bound to clip HU units on each RGB channel
            e.g. [(-1000, 0), (0, 1000), (-1000, 4000)]
        device: str
            'cuda' to run on GPU, 'cpu' to run on CPU
    """

    def __init__(self, key, windows, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.windows = windows
        self.key = key

        resnet = resnet50(pretrained=False)
        model = torch.nn.Sequential(*list(resnet.children())[:-2])

        # Load model
        url = 'https://github.com/owkin/scancovia/releases/download/v1.0/resnet50_moco_ct.pth.tar'
        ckpt = torch.hub.load_state_dict_from_url(url, progress=True, map_location=device)
        new_ckpt = OrderedDict((k[len('module.feature_extractor.'):], v)
                               for k, v in ckpt['state_dict'].items() if 'feature_extractor' in k)
        model.load_state_dict(new_ckpt)

        model.eval()
        model.to(device)
        self.model = model

    def __call__(self, sample):

        image = sample['image']  # shape (H, W, Z)
        # Grayscale to RGB
        image = image[None, ...].repeat(3, 0).transpose(
            (3, 0, 1, 2))  # shape (Z, C, H, W)
        # Clip HU and [0, 1] normalization
        windows = np.array(self.windows)
        vmin = windows[:, 0].reshape((1, 3, 1, 1))
        vmax = windows[:, 1].reshape((1, 3, 1, 1))
        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin)
        # Convert to Tensor
        image = torch.from_numpy(image).type(torch.FloatTensor).to(self.device)

        # Run model
        with torch.no_grad():
            features = self.model.forward(image).detach(
            ).cpu().numpy()  # shape (Z, 2048, H//32, W//32)
        sample.setdefault('features', {})[self.key] = features

        return sample


class ApplyPCA():
    """
    Applies a trained PCA on sample['feature'][key]

    Parameters
    ----------
        key: str
        mean: numpy array
        components: numpy array
    """

    def __init__(self, key, mean, components):
        self.key = key
        self.mean = mean
        self.components = components

    def __call__(self, sample):
        features = sample['features'][self.key]
        features -= self.mean.reshape(1, -1, 1, 1)
        features = np.tensordot(self.components, features, axes=(1, 1))
        features = features.transpose((1, 0, 2, 3))
        sample['features'][self.key] = features

        return sample


class ApplyLogisticRegression():
    """
    Applies a trained logistic regression on sample['feature'][key]

    Parameters
    ----------
        key: str
        coef: numpy array
        intercept: numpy array
    """

    def __init__(self, key, coef, intercept):
        self.name = key
        self.coef = coef
        self.intercept = intercept

    def __call__(self, sample):
        def sigmoid(x): return 1 / (1 + np.exp(-x))
        features = sample['features'][self.name]
        features = features.transpose((2, 3, 0, 1))  # shape (H, W, Z, C)
        heatmap = np.dot(features, self.coef) + self.intercept
        sample.setdefault('heatmap', {})[self.name] = heatmap
        sample.setdefault('prediction', {})[
            self.name] = sigmoid(heatmap.mean())
        return sample


class BlendPredictions():
    """
    Average predictions of different models
    """

    def __call__(self, sample):
        sample['prediction']['blend'] = np.mean(
            list(sample['prediction'].values()))
        sample['heatmap']['blend'] = np.mean(
            list(sample['heatmap'].values()), axis=0)
        return sample
