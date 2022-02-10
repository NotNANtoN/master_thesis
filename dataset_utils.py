import os

import torch
import pandas as pd
import numpy as np
from PIL import Image

from mimic_preprocessing import load_mimic_cxr


def load_dataset(dataset_name):
    os.makedirs(dataset_name, exist_ok=True)
    
    df_path = os.path.join(dataset_name, "final_df.pkl")
    label_path = os.path.join(dataset_name, "label_names.npy")
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
        label_names = np.load(label_path)
    else:
        if dataset_name == "mimic-cxr":
            df, label_names = load_mimic_cxr("/raid/datasets/mimic-cxr")
        df.to_pickle(df_path)
        np.save(label_path, label_names)
    return df, label_names
  