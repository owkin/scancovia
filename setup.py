from setuptools import setup, find_packages

# Only tested with python 3.7
setup(name='scancovia',
      version=0.1,
      author='OWKIN, INRIA',
      description='Code to run models of the ScanCovIa project',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.18.4',
          'pandas>=0.24.2',
          'SimpleITK>=1.2.4',
          'scikit-image>=0.15.0',
          'lungmask @ git+https://git@github.com/JoHof/lungmask@master',
          'efficientnet_pytorch @ git+https://git@github.com/lukemelas/EfficientNet-PyTorch@master',
          'torch>=1.4.0',
          'torchvision>=0.5.0',
          'nibabel>=2.4.1',
          'tqdm>=4.32.2',
          'scikit-learn>=0.23.1',
          'xgboost>=1.1.1',
          'shap>=0.34.0',
          #'tensorflow==1.14', #required for Liang benchmark
      ],
      long_description=open('README.md').read()
      )
