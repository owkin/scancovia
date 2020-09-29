# ScanCovIA 

This repository open sources all the AI-severity model presented in 
*Integration of clinical characteristics, lab tests and a deep learning CT scan
 analysis to predict severity of hospitalized COVID-19 patients* 
([link to the paper](https://www.medrxiv.org/content/10.1101/2020.05.14.20101972v1)), 
and reproduces alternative models presented in the literature.

## Available COVID-19 risk scores

AI-severity as well as 6 models trained on 646 patients from Hôpital Bicêtre AP-HP are open sourced:
- 4 models based on multimodal data:
   - **AI-severity** (main model)
   - AI-segment
   - Clinical and bio
   - Clinical and bio and radiologist report 
- 3 models based on radiology data only:
   - Radiologist report
   - AI-severity-ct
   - AI-segment-ct 

All these models output risk scores associated with patients diagnosed with COVID-19, based on different
data modalities. 

Associated class: `ScanCovModel` in `scan_cov_model.py`

### Benchmarking models


Eleven models from the literature are also reproduced:

- [CALL score](https://academic.oup.com/cid/advance-article-pdf/doi/10.1093/cid/ciaa414/33030013/ciaa414.pdf)
- [Colombi et al.](https://pubs.rsna.org/doi/pdf/10.1148/radiol.2020201433) (2 models)
- [COVID-GRAM](https://jamanetwork.com/journals/jamainternalmedicine/articlepdf/2766086/)
- [CURB-65](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1746657/pdf/v058p00377.pdf)
- [Yan et al.](https://www.nature.com/articles/s42256-020-0180-7.pdf)
- [MIT calculator](https://www.covidanalytics.io/mortality_calculator)
- [NEWS2](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2)
- [NEWS2 for COVID](https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v3.full.pdf+html)
- [4C score](https://www.bmj.com/content/370/bmj.m3339)
- [Liang et al.](https://www.nature.com/articles/s41467-020-17280-8)

Associated class: `Benchmarker` in `benchmarker.py`

### Performances of the models

AI-severity has been evaluated on 3 different outcomes and 2 cohorts:
- A cohort of 150 patients from Hôpital Bicêtre APHP (first row)
- An independent cohort of 135 patients from Gustave Roussy (second row)

![performances](./assets/performances.png?raw=True)

### Calibration
 
For the AI-severity model, the two terciles used to determine threshold values for low, medium and high risk groups were 
equal to 0.187, and 0.375.
![km](./assets/km_curves.png?raw=True)

## Installation

You can install this repository by running ```pip install git+https://github.com/owkin/scancovia.git```. 
This will install all the required dependencies defined in `setup.py`. Note that
tensorflow 1.14 is additionally required to run the Liang et al. benchmark

## Data format

All the models take as input a python dictionnary or a pandas DataFrame. The required variables for each model as well as their format are defined in the associated docstring in `scan_cov_model.py` and `benchmarks.py`.

Both AI-segment and AI-severity have been trained on thoracic CT-scans (parenchymal windowing). To run them, DICOM folders must be converted into NIfTI files - using for instance [dcm2niix](https://github.com/rordenlab/dcm2niix)) - and the path to these files must be provided in the input dictionnary/dataframe.


## Usage and examples

### AI severity

To run AI-severity on a single sample or on a dataset:
```
from scancovia import ScanCovModel
import pandas as pd

# Load the model
model = ScanCovModel()

# Run on a sample
sample = {
    'nii_path': 'patient_exam.nii',
    'Oxygen saturation': 80,
    'Age': 75,
    'Sex': 0, # 1 for Male, 0 for Female
    'Platelet': 4.5,
    'Urea': 1.3,
}
risk_sample = model(sample)

# Run on a dataset
#dataset = pd.read_csv('/path/to/your/dataset.csv')
#risk_dataset = model(dataset)
```

For the other models:

```
from scancovia import ScanCovModel
import pandas as pd

# Load the model (uncomment the model you want to use)
#name = 'AI-severity'
#name = 'AI-segment'
#name = 'Clinical and bio'
#name = 'Clinical and bio and radiologist report'
name = 'Radiologist report'
#name = 'AI-severity-ct'
#name = 'AI-segment-ct'

model = ScanCovModel(name)

# Run on a sample
sample = {
    # For Radiologist report only
    'Crazy paving': 0,
    'Topography': 1,
    'Predominance': 1,
    'Opacity': 0,
    'Condensation': 0,
    # For all multimodal models
    'Age': 62.,
    'Sex': 1,
    'Oxygen saturation': 95.,
    'Urea': 5.8,
    'Platelet': 208,
    # For some multimodal models
    'Chronic kidney disease': 0,
    'Disease extent': 2,
    'Neutrophil': 4.9,
    'LDH': 328.,
    'Hypertension': 0,
    'Dyspnea': 1,
    'Diastolic pressure': 80.,
    # For models with AI-segment or AI-severity
    'nii_path': 'patient_exam.nii',
}
risk_sample = model(sample)

# Run on a dataset
dataset = pd.read_csv('/path/to/your/dataset.csv')
risk_dataset = model(dataset)
```

It is also possible to run several models at the same time using `model = ScanCovModel([list of models])` or all models
 using `model = ScanCovModel('all')`.

_Note : models using AI-segment and AI-severity are much longer to run as they rely on deep learning algorithm. Usage of 
a GPU is higly recommended (see ScanCovModel docstring for more information)_

### AI-severity-ct

The AI-severity model takes as input the path to a NIfTI file and outputs an associated risk score

```
from scancovia import AiSeverity
ai_severity = AiSeverity()
risk = ai_severity('patient_exam.nii')
```

### AI-segment-ct

The AI-segment model takes as input the path to a NIfTI file and outputs a dictionnary containing the CT-scan (key='image'), a lung mask (key='lungs_mask'), and a lung lesion mask (key='segmentation_mask')

```
from scancovia import AiSegment
ai_segment = AiSegment() 
output = ai_segment('patient_exam.nii')
```

## Benchmarking models

The usage of the alternative models is very similar to the one of the ScanCov models:

```
from scancovia import Benchmarker

# Option 1: load a specific model (uncomment the model you want to use)
name = 'call'
#name = 'colombi'
#name = 'colombi_ct'
#name = 'covid_gram'
#name = 'curb_65'
#name = 'mit'
#name = 'yan_original'
#name = 'yan_retrained'
model_1 = Benchmarker(name)

# Option 2: load a list of models
model_2 = Benchmarker(['call', 'colombi'])

# Option 3: load all the models
model_3 = Benchmarker()
```

Then you can similarly run the model.s on a dictionnary or a dataframe

## Citation

This work is a joint work from Gustave Roussy, Hôpital Bicêtre AP-HP, Centre de Vision Numérique - Université Paris-Saclay - CentraleSupélec - Inria, and Owkin Inc. When using this repository, please consider citing the associated article:

```
@article {Lassau2020.05.14.20101972,
	author = {Lassau et al.},
	title = {Integration of clinical characteristics, lab tests and a deep learning CT scan analysis to predict severity of hospitalized COVID-19 patients},
	year = {2020},
	doi = {10.1101/2020.05.14.20101972},
	eprint = {https://www.medrxiv.org/content/early/2020/07/02/2020.05.14.20101972.full.pdf},
	journal = {medRxiv}
}
```
## License

This repository has been released under the GPL v3.0 license (see [LICENSE](LICENSE)). For any question relative to a commercial usage of the code please contact etienne.bendjebbar@owkin.com

## Medical disclaimer

This repository is for the purpose of disseminating health information free of charge for the benefit of the public and research-sharing purposes only and is made available on the basis that no professional advice on a particular matter is being provided.
Nothing contained in this repository is intended to be used as medical advice and it is not intended to be used to diagnose, treat, cure or prevent any disease, nor should it be used for therapeutic purposes or as a substitute for your own health professional’s advice. No liability is accepted for any injury, loss or damage incurred by use of or reliance on the information provided on this repository
