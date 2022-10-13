# Master thesis and AACL-IJCNLP Codebase

## Installation

To run, first install the requirements for python3.8 in the `requirements.txt`. 

Download MIMIC-CXR first (and Chexpert, covidx, and rsna for external validation).


## Data preprocessing

Run `preprocess_chexpert.ipynb`, `preprocess_covidx.ipynb`, `preprocess_rsna.ipynb`. Adjust the base path at the start to your download location. The path for MIMIC-CXR also needs to be adapted in `dataset_utils.py`in line 21 .

The `eda.ipynb` script can be used for some data exploration. 

## Training

The `contrastive_learning.ipynb` notebook takes care of the contrastive learning. The `sl.py` fine-tunes the whole network. The scripts `lin_probe_sklearn.py`and `lin_probe_multi.py` train linear probes on top of the CLIP features. 



## Evaluation

If you have trained all the models from the paper, the `eval_cl.ipynb` and `eval_sl.ipynb` notebooks will create all the plots and figures from the paper (except the text2image results).
