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

import pickle
import numpy as np
import pandas as pd
from math import log, exp
from functools import partial
from pathlib import Path

EPSILON = 0.01


def threshold_scorer(value, thresholds, scores):
    for t, s in zip(thresholds, scores):
        if value <= t:
            return s
    return scores[-1]


def call_score(sample):
    """
    Implementation of CALL score
    See Table 3 in https://academic.oup.com/cid/advance-article-pdf/doi/10.1093/cid/ciaa414/33030013/ciaa414.pdf

    Parameters
    ----------
    sample: dict
        Must contain the following keys:
        Age: int
        Comorbidity: bool or int. In the paper, comorbidity was defined as having at least 1 of the following: hypertension, 
        diabetes, cardiovascular disease, chronic lung disease, or human immunodeficiency virus infection for at least 6 months.
        In our manuscript, we defined Comorbidity as any('Cardiac disease', 'Asthma', 'Emphysema', 'Diabetes', 'Hypertension')
        LDH: int (in U/L)
        Lymphocyte: int (in g/L)

    Returns
    -------
    risk: float
    """

    risk = 1 + 3 * sample['Comorbidity']
    risk += 1 + 2 * (sample['Age'] > 60)
    risk += 1 + 2 * (sample['Lymphocyte'] <= 1)

    if sample['LDH'] <= 250:
        risk += 1
    elif 250 < sample['LDH'] <= 500:
        risk += 2
    else:
        risk += 3

    return risk


def colombi_score(sample, use_ct_scan=False):
    """
    Implementation of the model proposed in Colombi et al. (2020) RSNA
    See Table 3 in https://pubs.rsna.org/doi/pdf/10.1148/radiol.2020201433

    Parameters
    ----------
    sample: dict
        Must contain the following keys:
        Age: int
        Cardiovascular disease: bool or int. In our manuscript, we defined 'Cardiovascular disease' as
        any('Cardiac disease', 'Hypertension')
        Platelet: int (in g/L)
        CRP: int (in mg/L)
        LDH: int (in U/L)
        Disease extent:  int (0 absent,1 minimal (<10%),2 moderate (10-25%), 3 extensive (25-50%),
        4 severe (>50%), 5 critical (>75%))
    use_ct_scan: bool
        In Table 3, use weights from column multivariable clinical and CT if True,  multivariable clinical if False

    Returns
    -------
    risk: float
    """
    risk = 0

    if use_ct_scan:
        risk += 1.1 * (sample['Age'] > 68)
        risk += 1.4 * (sample['Cardiovascular disease'])
        risk += 1.1 * (sample['Platelet'] > 180)
        risk += 0.7 * ((sample['CRP'] * 0.1) > 7.6)  # mg / dL in the paper
        risk += 1.7 * (sample['Disease extent'] > 2)  # < 73% V-WAL in the paper
    else:
        risk += 1.2 * (sample['Age'] > 68)
        risk += 1.3 * (sample['Cardiovascular disease'])
        risk += 1.0 * (sample['Platelet'] > 180)
        risk += 1.1 * (sample['LDH'] > 347)
        risk += 1.0 * ((sample['CRP'] * 0.1) > 7.6)  # mg / dL in the paper

    # As no intercept is provided, we return the logit prediction
    return risk


def covid_gram(sample):
    """
    Implementation of COVID-Gram score
    See Table 3 in https://jamanetwork.com/journals/jamainternalmedicine/articlepdf/2766086/jamainternal_liang_2020_oi_200032.pdf

    Parameters
    ----------
    sample: dict
        Must contain the following keys:
        XRay abnormality: bool or int. In our manuscript we defined XRay abnormality as Disease extent != 0
        Age: int
        Hemoptysis: bool or int. In our manuscript, we set Hemoptysis to 0 as the variable was not available
        Dyspnea: bool or int
        Unconsciousness: bool or int. In our manuscript, we set Hemoptysis to 0 as the variable was not available
        Nb comorbidities gram: int (see Table 1). In our manuscript, we set 'Nb comorbidities gram' as sum('Cardiac disease',
        'Asthma', 'Diabetes', 'Hypertension', 'Emphysema', 'Chronic kidney disease')
        Cancer: bool or int
        Neutrophil: int (in g/L)
        Lymphocyte: int (in g/L)
        LDH: int (in U/L)
        Conjugated bilirubin: int (in umol/L)

    Returns
    -------
    risk: float
    """

    # Each coefficient is taken to be the log of the Odds Ratio (OR) reported in the paper
    risk = log(0.001)
    risk += log(3.39) * sample['XRay abnormality']
    risk += log(1.03) * sample['Age']
    risk += log(4.53) * sample['Hemoptysis']
    risk += log(1.88) * sample['Dyspnea']
    risk += log(4.71) * sample['Unconsciousness']
    risk += log(1.60) * sample['Nb comorbidities gram']
    risk += log(4.07) * sample['Cancer']
    risk += log(1.06) * sample['Neutrophil'] / (sample['Lymphocyte'] + EPSILON)
    risk += log(1.002) * sample['LDH']
    risk += log(1.15) * sample['Conjugated bilirubin']
    risk = 1 / (1 + exp(-risk))

    return risk


def curb_65(sample):
    """
    Implementation of CURB-65 score
    See Figure 2 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1746657/pdf/v058p00377.pdf

    Parameters
    ----------
    sample: dict
        Must contain the following keys:
        Confusion: bool or int. Defined as a Mental Test Score of 8 or less, or new disorientation in person, place or time.
        Urea: int (in mmol/L)
        Respiratory rate: int (in /min)
        Diastolic pressure: int (in mmHg)
        Systolic pressure: int (in mmHg)
        Age: int

    Returns
    -------
    risk: float

    """

    risk = 0
    risk += sample['Confusion']
    risk += sample['Urea'] > 7
    risk += sample['Respiratory rate'] >= 30
    risk += (sample['Diastolic pressure'] <= 60) or (sample['Systolic pressure'] < 90)
    risk += sample['Age'] >= 65

    return risk


def yan_score(sample, retrained=True):
    """
    Implementation of Nature MI Score
    See https://www.nature.com/articles/s42256-020-0180-7.pdf
    Instead of using the tree in Figure 2 which outputs binary output and provided poor results
    we retrained it on our cohort

    Parameters
    ----------
    sample: dict
        Must contain the following keys
        LDH: int (in U/L)
        CRP: int (in mg/L)
        Pct lymphocytes: float (percentage of lymphocytes in %)
    retrained: bool
        Whether to use the original model of the paper or the one retrained on the manuscript cohort

    Returns
    -------
    risk: float
    """

    if retrained:
        if (sample['LDH'] < 378.5) or np.isnan(sample['LDH']):
            if (sample['CRP'] < 90.5) or np.isnan(sample['CRP']):
                risk = -0.3896
            else:
                risk = -0.1348
        else:
            if (sample['Pct lymphocytes'] < 9.32) or np.isnan(sample['Pct lymphocytes']):
                risk = -0.1176
            else:
                risk = 0.1818
        risk = 1 / (1 + exp(-risk))
    else:
        # No information is provided on how to deal with missing values
        risk = 0
        if (sample['LDH'] < 365):
            if (sample['CRP'] < 41.2) or (sample['Pct lymphocytes'] > 14.7):
                risk = 1

    return risk


def get_mit_score():
    with open(Path(__file__).parents[1] / 'assets/model_without_lab.pkl', 'rb') as file:
        pkl = pickle.load(file)
        classifier = pkl['model']
        imputer = pkl['imputer']

    def mit_score(sample):
        """
        Implementation of the MIT analytics calculator
        See https://www.covidanalytics.io/mortality_calculator and https://github.com/COVIDAnalytics/website

        Parameters
        ----------
        sample: dict
            Must contain the following keys:
            Age: int
            Body temperature: int (in celsius degrees)
            Cardiac frequency: int (in /min)
            Cardiac dysrhythmias: bool or int
            In our manuscript we used the KNN imputer provided by the model
            Chronic kidney disease: bool or int
            Cardiovascular disease: bool or int
            In our manuscript we used any('Cardiac disease', 'Hypertension')
            Diabetes: bool or int
            Sex: bool or int (1 for male, 0 for female)
            Oxygen saturation: int (in %)

        Returns
        -------
        risk: float
        """

        df_sample = pd.DataFrame({'Age': [sample['Age']],
                                  'Body Temperature': [sample['Body temperature']],
                                  'Cardiac Frequency': [sample['Cardiac frequency']],
                                  'Cardiac dysrhythmias': [sample['Cardiac dysrhythmias']],
                                  'Chronic kidney disease': [sample['Chronic kidney disease']],
                                  'Coronary atherosclerosis and other heart disease': [
                                      sample['Cardiovascular disease']],
                                  'Diabetes': [sample['Diabetes']],
                                  'Gender': [1 - sample['Sex']],
                                  'SaO2': [sample['Oxygen saturation']]
                                  })
        df_sample[:] = imputer.transform(df_sample)
        risk = classifier.predict_proba(df_sample)[0, 1]
        return risk

    return mit_score


def news2(sample):
    """
    Implementation of NEWS2 score
    Source: https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2
    Scale 1 scores are used for Oxygen saturation

    Parameters
    ----------
    sample: dict
        Must contain the following keys:
        Respiratory rate: int (in /min)
        Oxygen saturation: int (in %)
        Air or oxygen: int (0 for Air, 1 for oxygen)
        Systolic pressure: int (in mmHg)
        Cardiac frequency: int (in /min)
        Unconsciousness: bool or int (True if unconscious)
        Body temperature: int (in celsius degrees)

    Returns
    -------
    risk: float
    """

    risk = 0
    risk += threshold_scorer(sample['Respiratory rate'], [8, 11, 20, 24], [3, 1, 0, 2, 3])
    risk += threshold_scorer(sample['Oxygen saturation'], [91, 93, 95], [3, 2, 1, 0])
    risk += threshold_scorer(sample['Air or oxygen'], [0.5], [0, 2])
    risk += threshold_scorer(sample['Systolic pressure'], [90, 100, 110, 219], [3, 2, 1, 0, 3])
    risk += threshold_scorer(sample['Cardiac frequency'], [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    risk += threshold_scorer(sample['Unconsciousness'], [0.5], [0, 3])
    risk += threshold_scorer(sample['Body temperature'], [35, 36, 38, 39], [3, 1, 0, 1, 2])

    return risk


def news2_carr(sample):
    """
    Implementation of NEWS2 updated score
    Source: https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v3.full.pdf+html
    Weights from: https://github.com/ewancarr/NEWS2-COVID-19 (model 4)

    Parameters
    ----------
    sample: dict
        Must contain the following keys:
        (for NEWS2)
        Respiratory rate: int (in /min)
        Oxygen saturation: int (in %)
        Air or oxygen: int (0 for Air, 1 for oxygen)
        Systolic pressure: int (in mmHg)
        Cardiac frequency: int (in /min)
        Unconsciousness: bool or int (True if unconscious)
        Body temperature: int (in celsius degrees)
        (for the updated score)
        Oxygen liters: int (in L/min)
        Urea: int (in mmol/L)
        Age: int
        CRP: int (in mg/L)
        EGFR: int (Estimated Glomerular Filtration Rate in mL/min)
        Neutrophil: int (in g/L)
        Lymphocyte: int (in g/L)

    Returns
    -------
    risk: float
    """

    sample['news2'] = news2(sample)
    sample['nlr'] = sample['Neutrophil'] / (sample['Lymphocyte'] + EPSILON)

    risk = -0.9687
    risk += 0.1450 * (np.sqrt(sample['CRP']) - 9.1433) / 4.1398
    risk += -0.2320 * (sample['EGFR'] - 60.4115) / 25.8918
    risk += 0.0887 * (np.sqrt(sample['Neutrophil']) - 2.4072) / 0.6980
    risk += 0.0577 * (np.sqrt(sample['Urea']) - 2.8909) / 1.0338
    risk += 0.0664 * (np.log(sample['nlr'] + EPSILON) - 1.7255) / 0.7520
    risk += -0.1987 * (sample['Oxygen saturation'] - 95.9230) / 2.5935
    risk += 0.3295 * (sample['Oxygen liters'] - 3.1818) / 4.9347
    risk += 0.4171 * (sample['news2'] - 2.9231) / 2.4468
    risk += 0.2795 * (sample['Age'] - 69.2258) / 16.7558
    risk = 0.9689 * risk - 0.0232  # calibration
    risk = 1 / (1 + np.exp(-risk))

    return risk


def score_4C(sample):
    """
    Implementation of the 4C mortality score
    Source: https://www.bmj.com/content/370/bmj.m3339, Table 2

    Parameters
    ----------
    sample: dict
        Must contain the following keys:
        Age: int (in years)
        Sex: bool or int (1 for male, 0 for female)
        Nb comorbidities 4C: int. In our manuscript, we set 'Nb comorbidities'
        as sum('Cardiac disease', 'Diabetes', 'Emphysema', 'Chronic kidney disease','Cancer')
        Respiratory rate: int (in /min)
        Oxygen saturation: int (in %)
        Glasgow coma score: int. In our manuscript, we used a constant value
        Urea: int (in mmol/L)
        CRP: int (in mg/L)

    Returns
    -------
    risk: float
    """

    risk = 0
    risk += threshold_scorer(sample['Age'], [49, 59, 69, 79], [0, 2, 4, 5, 7])
    risk += sample['Sex']
    risk += threshold_scorer(sample['Nb comorbidities 4C'], [0, 1], [0, 1, 2])
    risk += threshold_scorer(sample['Respiratory rate'], [19, 29], [0, 1, 2])
    risk += 2 * (sample['Oxygen saturation'] < 92)
    risk += 2 * (sample['Glasgow coma score'] < 15)
    risk += threshold_scorer(sample['Urea'], [7, 14], [0, 1, 3])
    risk += threshold_scorer(sample['CRP'], [50, 99], [0, 1, 2])

    return risk


def get_liang():
    from scancovia.liang_utils import Model_COX_DL, Model_NFold, MODEL_PATH  # for tensorflow
    model_dl = Model_NFold(MODEL_PATH)
    model_cox = Model_COX_DL()

    def liang(sample):
        """
        Implementation of https://www.nature.com/articles/s41467-020-17280-8
        Weights from: https://github.com/cojocchen/covid19_critically_ill
        Online model: https://aihealthcare.tencent.com/COVID19-Triage_en.html

        Parameters
        ----------
        sample: dict
            Must contain the following keys:
            Nb comorbidities liang: int. In our manuscript, we set 'Nb comorbidities liang'
            as sum('Diabetes', 'Hypertension', 'Cardiac disease', 'Chronic kidney disease',
            'Cancer', 'Emphysema')
            LDH: int (in U/L)
            Age: int (in years)
            Neutrophil: int (in g/L)
            Lymphocyte: int (in g/L)
            Creatine kinase: int (in U/L)
            Conjugated bilirubin: int (in umol/L)
            Cancer: bool or int
            Chronic obstructive pulmonary disease: bool or int. In our manuscript we used Emphysema
            XRay abnormality: bool or int. In our manuscript we defined XRay abnormality as Disease extent != 0
            Dyspnea: bool or int

        Returns
        -------
        risk: float
        """

        new_sample = {
            'Number.of.comorbidities': sample['Nb comorbidities liang'],
            'Lactate.dehydrogenase': sample['LDH'],
            'Age': sample['Age'],
            'NLR': sample['Neutrophil'] / (sample['Lymphocyte'] + EPSILON),
            'Creatine.kinase': sample['Creatine kinase'],
            'Direct.bilirubin': sample['Conjugated bilirubin'],
            'Malignancy': sample['Cancer'],
            'X.ray.abnormality': sample['XRay abnormality'],
            'COPD': sample['Chronic obstructive pulmonary disease'],
            'Dyspnea': sample['Dyspnea']
        }
        new_sample = dict((k, float(v)) for k, v in new_sample.items())

        new_sample['DL.feature'] = model_dl.predict(new_sample)[0, 0]
        risk = model_cox.predict(new_sample)['score']
        return risk

    return liang


class Benchmarker:
    """
    Alternative model_list to the one proposed in the ScanCovModel class
    As the ScanCovModel, most of the model_list are not calibrated

    Parameters
    ----------
    model_list: tuple of str
        list of model_list to use. By default, all models are used. Available models are in self.available_models
    """

    available_models = {
        'call': lambda: call_score,
        'colombi': lambda: partial(colombi_score, use_ct_scan=False),
        'colombi_ct': lambda: partial(colombi_score, use_ct_scan=True),
        'covid_gram': lambda: covid_gram,
        'curb_65': lambda: curb_65,
        'mit': get_mit_score,
        'yan_original': lambda: partial(yan_score, retrained=False),
        'yan_retrained': lambda: partial(yan_score, retrained=True),
        'news_2': lambda: news2,
        'news2_carr': lambda: news2_carr,
        'score_4C': lambda: score_4C,
        'liang': get_liang,
    }

    def __init__(self, model_list=None):
        # Check input
        if model_list is not None:
            if isinstance(model_list, str):
                model_list = [model_list]
            else:
                assert isinstance(model_list, list)
            assert all([model_name in self.available_models for model_name in model_list])
            self.model_list = model_list
        else:
            self.model_list = list(self.available_models.keys())
            
        self.initialized_models = dict((model_name, self.available_models[model_name]())
                                       for model_name in self.model_list)

    def __call__(self, sample_or_dataset):
        """
        Computes the risk scores for the different benchmarks

        Parameters
        ----------
        sample_or_dataset: dict or pandas DataFrame
            Must contain the mandatory keys/columns for each model in self.model_list
            These keys/columns are precised in the docstring of each model

        Returns
        -------
        df_scores: same type as sample_or_dataset
            If sample_or_dataset is a dict, returns a dict with keys=models
            If sample_or_dataset is a DataFrame, returns a DataFrame with columns=models and rows=samples  
        """

        # Check inputs
        assert type(sample_or_dataset) in [dict, pd.DataFrame]
        if isinstance(sample_or_dataset, dict):
            dataset = pd.DataFrame(dict((k, [v]) for k, v in sample_or_dataset.items()))
        else:
            dataset = sample_or_dataset

        # Run models
        df_scores = pd.DataFrame()
        for idx, row in dataset.iterrows():
            for model_name in self.model_list:
                sample = dict(row).copy()
                df_scores.at[idx, model_name] = self.initialized_models[model_name](sample)

        # Return risk score.s
        if isinstance(sample_or_dataset, dict):
            return dict(df_scores.iloc[0])
        else:
            return df_scores
