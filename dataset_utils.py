import os

import torch
import pandas as pd
import numpy as np
from PIL import Image

from mimic_preprocessing import load_mimic_cxr


def load_dataset(dataset_name, chexpert_label_subset=True):
    os.makedirs(dataset_name, exist_ok=True)
    
    df_path = os.path.join(dataset_name, "final_df.pkl")
    label_path = os.path.join(dataset_name, "label_names.npy")
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
        label_names = np.load(label_path)
    else:
        if dataset_name == "mimic-cxr":
            df, label_names = load_mimic_cxr("/raid/datasets/mimic-cxr")
        else:
            print("Run the preprocessing script first!")
        df.to_pickle(df_path)
        np.save(label_path, label_names)
        
    # sel
    if dataset_name == "chexpert" and chexpert_label_subset:
        selected = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        # get mapping
        idcs_in_orig = [i for i, l in enumerate(label_names) if l in selected]
        # fix label names
        label_names = selected
        # fix actual labels
        stacked_labels = np.stack(df["labels"].to_numpy())
        subset_labels = stacked_labels[:, idcs_in_orig]
        df["labels"] = [l for l in subset_labels]
        
        
    """
    # make custom split for mimic-cxr
    if dataset_name == "mimic-cxr":
        from sklearn.model_selection import train_test_split
        labels = df["labels"].to_numpy()
        # fill -1 and nan labels
        labels[labels == -1] = 0
        labels = np.nan_to_num(labels)
        # turn labels into strings for stratification
        print("Label proportions all data: ", labels.mean(axis=0))

        from skmultilearn.model_selection import iterative_train_test_split
        dev_df, y_dev, test_df, y_test = iterative_train_test_split(df, labels, test_size=0.1)
    """

        
    return df, label_names
  