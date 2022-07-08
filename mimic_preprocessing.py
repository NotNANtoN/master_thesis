import os
from pathlib import Path
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def load_mimic_cxr(path):
    # create output folder
    os.makedirs("mimic-cxr", exist_ok=True)
    mimic_cxr_path = Path(path)
    # read df
    df = pd.read_csv(mimic_cxr_path / 'cxr-record-list.csv.gz', header=0, sep=',')
    # format image paths
    df["img_path"] = df["path"].apply(lambda x: (str((mimic_cxr_path / 'jpg') / (x[:-4] + '.jpg'))).replace("/files/", "/small_files/"))
    # add captions
    df = add_captions(df, mimic_cxr_path)
    # preprocess captions
    preprocessed_caps = []
    print("Preprocessing captions...")
    for cap in tqdm(df["caption"]):
        cap = preprocess_cap(cap)
        preprocessed_caps.append(cap)
    df["prepro_caption"] = preprocessed_caps
    # load split
    split_df = load_mimic_split(mimic_cxr_path)
    df = df.merge(split_df[["dicom_id", "split"]], on="dicom_id", how="outer")
    # load labels
    chexpert_label_df, label_names = load_mimic_labels(mimic_cxr_path)
    df = df.merge(chexpert_label_df[["study_id", "labels"]], on="study_id", how="outer")
    # drop where labels are NaN
    df = df.dropna(subset=["labels"])
    # rename stuff to match other datasets
    df = df.rename(columns={"dicom_id": "img_id",
                            "prepro_caption": "caption",
                            "caption": "raw_caption"})
    if "path" in df:
        df = df.drop(columns=["path"])
                          
    #  downsize images to around 256x256 (original res is something like 3kx2k)
    print("Downsizing images to 256 resolution - this takes a while (probably 12-24 hours)...")
    results = Parallel(n_jobs=16, verbose=1)(
                 map(delayed(resize_and_save), df["img_path"].to_list()))
                    
    return df, label_names


def show_mimic_content(used_df, idx=0):
    print("Using index: ", idx)
    cap = used_df["prepro_caption"].iloc[idx]
    raw_cap = used_df["caption"].iloc[idx]
    print(raw_cap)
    print()
    print(cap)
    print("Num words of preprocessed caption: ", len(cap.split(" ")))

    ax = plt.figure(figsize=[16, 10])
    plt.imshow(Image.open(used_df["img_path"].iloc[idx]), cmap="gray")
    plt.axis("off")
    plt.show()
#show_mimic_content(df[final_adds], 20)


def load_mimic_split(mimic_cxr_path):
    # load train/val/test splits
    split_df = pd.read_csv(mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz', header=0, sep=',')
    print("Counts: ", np.unique(split_df["split"], return_counts=True))
    return split_df    


def load_mimic_labels(mimic_cxr_path):
    # load labels
    #negbio_label_df = pd.read_csv(mimic_cxr_path / 'mimic-cxr-2.0.0-negbio.csv.gz', header=0, sep=',')
    chexpert_label_df = pd.read_csv(mimic_cxr_path / 'mimic-cxr-2.0.0-chexpert.csv.gz', header=0, sep=',')

    label_names = ['Atelectasis', 'Cardiomegaly',
           'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
           'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
           'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    chexpert_label_df["labels"] = chexpert_label_df.apply(lambda x: x[label_names].to_numpy(), axis=1)
    return chexpert_label_df, label_names


def load_mimic_caption(path):
    # load captions from files
    with open(path, 'r') as fp:
        text = "".join(fp.readlines())
    return text


def add_captions(df, mimic_cxr_path):
    # WARNING: takes relatively long, hence the save
    temp_path = "mimic-cxr/df_with_captions.csv"
    if os.path.exists(temp_path):
        df = pd.read_csv(temp_path, index_col=0)
    else:
        print("Reading in all captions from disk and saving them to a dataframe... Takes approx. 12 hours")
        df["caption"] =  df["path"].apply(lambda x: load_mimic_caption(mimic_cxr_path / "reports/" / ("/".join(x.split("/")[:-1]) + '.txt')))
        df.to_csv(temp_path)
    return df


def preprocess_cap(cap):
    sections_to_remove = ["TECHNIQUE", "COMPARISON", "DATE", "EXAMINATION"]
    for sec in sections_to_remove:
        #print(sec)
        cap = cap.replace(sec + "\n", sec).replace(sec + " \n", sec)
        while sec in cap:
            start_idx = cap.find(sec)
            end_idx = cap[start_idx:].find("\n") 
            if end_idx == -1:
                break
            end_idx += start_idx
            cap = cap[:start_idx] + cap[end_idx:]
            #print("---")
            #print(cap)
            #print(start_idx, end_idx)
            #print("---")
    cap = cap.replace("FINAL REPORT", " ")
    cap = cap.strip().replace("\n \n", "\n").replace("\n \n", "\ņ").replace(".\n", ".").replace("\n", " ").replace("\ņ", "")
    while "__" in cap:
        cap = cap.replace("__", "_")
    cap = cap.replace("CHEST RADIOGRAPH PERFORMED ON _", "").replace("CHEST RADIOGRAPH", "")
    cap = cap.replace("_ year old", "").replace("_ years old", "").replace("_-year-old", "")
    while "  " in cap:
        cap = cap.replace("  ", " ")
    cap = cap.replace("TYPE OF EXAMINATION", "EXAMINATION")
    cap = cap.replace("FINAL ADDENDUM ADDENDUM", "FINAL ADDENDUM")
    
    # filter comments? - at least kill the repetitions within
    # kill repeats (happens mostly in comments)!

    return cap.strip()


def resize_img(img, basewidth=256):
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    return img.resize((basewidth,hsize), Image.LANCZOS)


def resize_and_save(path):
    # create paths
    new_path = path.replace("/files/", "/small_files/")
    folder_name = "/".join(new_path.split("/")[:-1])
    os.makedirs(folder_name, exist_ok=True)
    # downsample
    small_img = resize_img(Image.open(path), basewidth=256)
    # save
    small_img.save(new_path, subsample=0, quality=95)
