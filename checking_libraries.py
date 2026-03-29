import sys
print('='*55)
print(f'Python      : {sys.version}')
print('='*55)

import torch
print(f'PyTorch     : {torch.__version__}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU name    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM total  : {round(torch.cuda.get_device_properties(0).total_memory/1e9,2)} GB')
else:
    print('GPU name    : CPU ONLY — CUDA not detected')
print('-'*55)

import numpy;      print(f'NumPy       : {numpy.__version__}')
import pandas;     print(f'Pandas      : {pandas.__version__}')
import scipy;      print(f'SciPy       : {scipy.__version__}')
print('-'*55)

import sklearn;    print(f'scikit-learn: {sklearn.__version__}')
import xgboost;    print(f'XGBoost     : {xgboost.__version__}')
import shap;       print(f'SHAP        : {shap.__version__}')
print('-'*55)

import matplotlib; print(f'Matplotlib  : {matplotlib.__version__}')
import seaborn;    print(f'Seaborn     : {seaborn.__version__}')
import plotly;     print(f'Plotly      : {plotly.__version__}')

try:
    import kaleido
    ver = getattr(kaleido, '__version__', 'installed (v0.4+)')
    print(f'Kaleido     : {ver}')
except ImportError:
    print('Kaleido     : NOT INSTALLED  <- pip install kaleido')


import flask;      print(f'Flask       : {flask.__version__}')
try:
    # import flask_cors
    # print(f'Flask-CORS  : {flask_cors.__version__}')
    import importlib.metadata
    flask_ver = importlib.metadata.version("flask")
    print(f'Flask       : {flask_ver}')

except ImportError:
    print('Flask-CORS  : NOT INSTALLED  <- pip install flask-cors')
print('-'*55)

import tqdm;       print(f'tqdm        : {tqdm.__version__}')
import joblib;     print(f'joblib      : {joblib.__version__}')
import tabulate;   print(f'tabulate    : {tabulate.__version__}')
try:
    import dotenv
    print(f'python-dotenv: installed')
except ImportError:
    print('python-dotenv: NOT INSTALLED  <- pip install python-dotenv')

print('='*55)
print('VERIFICATION COMPLETE')
print('='*55)
