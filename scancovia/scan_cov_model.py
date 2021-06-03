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
import pandas as pd
import torch
from tqdm import tqdm

from scancovia.ai_severity.ai_severity import AiSeverity
from scancovia.ai_segment.ai_segment import AiSegment

# Weights of the logistic regressions
WEIGHTS = {
    "Radiologist report": {
        "Opacity": -0.67385675,
        "Crazy paving": 0.40283637,
        "Condensation": -0.0878739,
        "Topography": -0.13998417,
        "Predominance": -0.45379975,
        "Disease extent": 0.6869313,
        "intercept": -1.72478553,
    },
    "AI-segment-ct": {
        "GGO AiSegment": -1.7286270941282955,
        "Consolidation AiSegment": 3.2828752888604242,
        "Crazy paving AiSegment": 2.9325930870155235,
        "Disease extent AiSegment": 4.485950268831,
        "intercept": -1.861267060140949,
    },
    "AI-severity-ct": {
        "AiSeverity": 1,
        "intercept": 0,
    },
    "Clinical and bio": {
        "Oxygen saturation": -0.7734987379380287,
        "Age": 0.01801136309055773,
        "Sex": 0.5142519676707926,
        "LDH": 1.1940053678318288,
        "Platelet": -1.158777591461933,
        "Chronic kidney disease": 0.26417295778519984,
        "Dyspnea": 0.3195490108875548,
        "Hypertension": 0.19583033430499872,
        "Neutrophil": 0.5041151224479017,
        "Urea": 0.3927025448161429,
        "intercept": -6.633691346366725,
    },
    "Clinical and bio and radiologist report": {
        "Oxygen saturation": -0.715867705052558,
        "Disease extent": 0.6344741735315738,
        "Age": 0.023845726941254343,
        "Sex": 0.5481826956249627,
        "Platelet": -0.8337314547539033,
        "Urea": 0.5606860212992523,
        "intercept": -2.3138923947470724,
    },
    "AI-severity": {
        "Oxygen saturation": -0.5691738927009029,
        "AiSeverity": 0.768643675292477,
        "Age": 0.01208368240782795,
        "Sex": 0.4116879766154334,
        "Platelet": -0.5667267219464422,
        "Urea": 0.393194190948113,
        "intercept": -0.05311807245634939,
    },
    "AI-segment": {
        "Oxygen saturation": -0.6807632355815612,
        "Disease_extent_both": 3.5406023074928723,
        "Age": 0.02086328048247559,
        "Sex": 0.6924592728262375,
        "Platelet": -0.9178319465213124,
        "Urea": 0.548335781823739,
        "intercept": -1.0383591562158774,
    },
}

# Preprocessing transforms
TRANSFORMS = {
    name: lambda x: np.log(0.001 + x)
    for name in [
        "Urea",
        "Neutrophil",
        "LDH",
        "Platelet",
    ]
}
TRANSFORMS["Oxygen saturation"] = lambda x: -np.log(1 + 100 - x)


class ScanCovModel():
    """
    The ScanCovModel class computes COVID-19 risk scores for 7 different models :

    Radiological score:
        - 'Radiologist report' : CT scans annotations of radiologists only
        - 'AI-segment-ct' : automatic CT scans annotations only
        - 'AI-severity-ct' : automatic CT scans risk score only

    Multimodal scores:
        - 'Clinical and bio' : clinical and biology data only
        - 'Clinical and bio and radiologist report' : CT scans annotations of radiologists, clinical and biology data
        - 'AI-segment' : automatic CT scans annotations, clinical and bio data
        - 'AI-severity' : automatic CT scans risk score, clinical and bio data

    Input variables are accessible through the attribute input_variables
    The main model is AI-severity

    Models involving AI-segment or AI-severity are much longer to run as they involved deep neural network
    On a NVIDIA Tesla P40, AI-severity requires less than 10sec to run, and AI-segment around 1min 
    On a 64 cores machine, AI-severity requires around 1min to run and AI-segment around 15min

    Parameters
    ----------
    model_list: str or list of str
        model name or list of model names. 'all' for all models
        By default, only the ScanCov score ('AI-severity') is computed
    device: str
        Device on which run AI-segment or AI-severity when necessary
        'cuda' to run on GPUs, 'cpu' to run on CPU

    Examples
    --------
    >>> from scancovia import ScanCovModel
    >>> model = ScanCovModel()
    >>> sample = {
    >>>    "Oxygen saturation": 80,
    >>>    "Disease extent": 4,
    >>>    "Age": 75,
    >>>    "Sex": 0,
    >>>    "Platelet": 4.5,
    >>>    "Urea": 1.3,
    >>> }
    >>> risk = model(sample)
    >>> # see README.md for more examples
    """

    available_models = ['Radiologist report',
                        'AI-segment-ct',
                        'AI-severity-ct',
                        'Clinical and bio',
                        'Clinical and bio and radiologist report',
                        'AI-severity',
                        'AI-segment']

    def __init__(self, model_list='AI-severity', device=None):

        # Check model_list input
        assert type(model_list) in [str, list]
        message = f'Available models : {self.available_models}'
        if isinstance(model_list, str):
            if model_list == 'all':
                self.model_list = self.available_models
            else:
                assert model_list in self.available_models, message
                self.model_list = [model_list]
        else:
            assert all([model_name in self.available_models for model_name in model_list]), message
            self.model_list = model_list

        # Load AI-segment or AI-severity if necessary
        self.ai_segment = None
        self.ai_severity = None

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            assert device in ['cuda', 'cpu']
        self.device = device
        self.input_variables = set(sum([list(WEIGHTS[model_name].keys()) for model_name in self.model_list], []))
        self.input_variables.remove('intercept')

        if any(['segment' in model_name for model_name in self.model_list]):
            self.ai_segment = AiSegment(device)
            self.input_variables.add('nii_path')

        if any(['severity' in model_name for model_name in self.model_list]):
            self.ai_severity = AiSeverity(device)
            self.input_variables.add('nii_path')

    def __call__(self, sample_or_dataset):
        """
        Run the model.s on a sample or a dataset

        Parameters
        ----------
        sample_or_dataset: dict or pandas DataFrame
            Depending on the model (see WEIGHTS variables), must contain the following keys/columns:
            # Clinical
            Age: int
            Sex: bool or int (1 for male, 0 for female)
            Oxygen saturation: int (in %)
            Diastolic pressure: int (in mmHg)
            Hypertension: bool or int
            Chronic kidney disease: bool or int
            Dyspnea: bool or int
            # Biological
            Platelet : int (in g/L)
            Neutrophil: int (in g/L)
            Urea : int (in mmol/L)
            LDH : int (in U/L)
            # Radiological
            nii_path: str (path to the NIfTI file for models using AI-segment or AI-severity)
            Disease extent:  int (0 absent,1 minimal (<10%),2 moderate (10-25%), 3 extensive (25-50%),
                    4 severe (>50%), 5 critical (>75%))
            Opacity: bool or int (1 if yes else 0)
            Crazy paving: bool or int (1 if yes else 0)
            Condensation: bool or int (1 if yes else 0)
            Topography: bool or int (1 if subpleural else 0)
            Predominance: bool or int (1 if yes else 0)

        Returns
        -------
        df_scores: same type as sample_or_dataset
            If sample_or_dataset is a dict, returns a dict with keys = models
            If sample_or_dataset is a DataFrame, returns a DataFrame with columns=models and rows=samples            
        """

        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        logit = lambda x: np.log(x / (1 - x))

        # Check inputs
        assert type(sample_or_dataset) in [dict, pd.DataFrame]
        iterator = iter
        if isinstance(sample_or_dataset, dict):
            dataset = pd.DataFrame(dict((k, [v]) for k, v in sample_or_dataset.items()))
        else:
            dataset = sample_or_dataset
            if (self.ai_segment is not None) or (self.ai_severity is not None):
                # Use tqdm when running inference on a dataset with AI-segment or AI-severity
                iterator = tqdm

                # Run models
        df_scores = pd.DataFrame()
        for idx, row in iterator(dataset.iterrows()):
            sample = dict(row).copy()
            # Run AI-segment or AI-severity when necessary
            if self.ai_segment is not None:
                sample.update(self.ai_segment(sample['nii_path']))
            if self.ai_severity is not None:
                sample['AiSeverity'] = logit(self.ai_severity(sample['nii_path']))
            # Apply logistic regression for each model
            for model_name in self.model_list:
                risk = WEIGHTS[model_name]['intercept']
                for key, coef in WEIGHTS[model_name].items():
                    if key != 'intercept':
                        risk += coef * TRANSFORMS.get(key, lambda x: x)(sample[key])
                risk = sigmoid(risk)
                df_scores.at[idx, model_name] = risk

        # Return risk score.s
        if isinstance(sample_or_dataset, dict):
            return dict(df_scores.iloc[0])
        else:
            return df_scores
