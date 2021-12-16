import os
import string

import torch
from PIL import Image
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')



class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        super().__init__()
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]))
    


def filter_words(sent):
    stop_words = set(stopwords.words('english'))
    #stop_words.remove("of")
    for w in ["other", "scene", "notvisual", "one", "something", "foreground", "background", 
              "people", "many", "several", "bunch", "lot", "left", "right", "top", "bottom", "others", "another"]:
        stop_words.add(w)
    word_tokens = word_tokenize(sent)
    filtered_sentence = [w.lower().translate(string.punctuation) for w in word_tokens]
    filtered_sentence = [w for w in filtered_sentence if w not in stop_words and w.isalpha()]
    return " ".join(filtered_sentence)


def get_valid_key_names(inv_dict, k=10, verbose=False, filter_words=None):
    sorted_keys = sorted(inv_dict, key=lambda x: len(set([l // 5 for l in inv_dict[x]])), reverse=True)
    valid_label_names = []
    count = 0
    for key in sorted_keys:
        if filter_words is not None:
            if key in filter_words:
                continue
        if verbose:
            print(key)
            print(len(inv_dict[key]))
        if len(key) > 2:
            valid_label_names.append(key)
        count += 1
        if count == k:
            break
    return valid_label_names


def get_labels(sent_data):
    labels = [[phrase["phrase"] for phrase in sent_dict["phrases"]] for sent_dict in sent_data]
    labels += [[phrase["phrase_type"] for phrase in sent_dict["phrases"]] for sent_dict in sent_data]
    flat_labels = set()
    for label in labels:
        if isinstance(label, list):
            for l in label:
                if isinstance(l, list):
                    for lab in l:
                        flat_labels.add(lab)
                else:
                    flat_labels.add(l)
        else:
            flat_labels.add(label)
    flat_labels = list(flat_labels)
    labels = sorted([element for element in set([filter_words(sent) for sent in flat_labels]) if len(element) > 2])
    return labels

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# load img paths
def is_img(img_name):
    return img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png")


def load_flickr(path, num_imgs=None):
    img_folder = os.path.join(path, "images")
    imgs = [i for i in os.listdir(img_folder) if is_img(i)]
    imgs = sorted(imgs, key = lambda x: int(x.split(".")[0]))
    img_ids = [int(x.split(".")[0]) for x in imgs]
    img_paths = [os.path.join(img_folder, i) for i in imgs]

    # load captions
    captions = pd.read_csv(os.path.join(path, "captions.txt"), delimiter="|")
    captions["img_idx"] = captions["image_name"].apply(lambda x: int(x.split(".")[0]))
    captions = captions.sort_values("img_idx")
    del captions["img_idx"]
    
    if num_imgs is not None:
        imgs = imgs[:num_imgs]
        img_paths = imgs[:num_imgs]
        img_ids = img_ids[:num_imgs]
        captions = captions[:num_imgs * 5]
    
    # annotated bounding box labels:
    # from https://github.com/BryanPlummer/flickr30k_entities
    import sys
    sys.path.append("/raid/datasets")
    from flickr30k_entities.flickr30k_entities_utils import get_annotations, get_sentence_data
    from collections import defaultdict

    labels = []
    label_count = defaultdict(int)

    # load relations between img bounding boxes and captions
    for img_id in tqdm(img_ids):
        #p = f"/raid/datasets/flickr30k_entities/Annotations/{img_id}.xml"
        #annot = get_annotations(p)
        p = f"/raid/datasets/flickr30k_entities/Sentences/{img_id}.txt"
        sent_data = get_sentence_data(p)
        img_labels = get_labels(sent_data)
        # increase counts
        for l in img_labels:
            label_count[l] += 1
        # store
        labels.append(img_labels)

    # get label names with highest count
    top_k = 200
    label_series = pd.Series(label_count.values(), index=label_count.keys()).sort_values(ascending=False).iloc[:top_k]
    label_names = list(label_series.index)
    # sort through all labels to only keep high count labels
    labels = [[l for l in label_list if l in label_names] for label_list in labels]

    figsize = (6, 50)
    plt.figure(figsize=figsize)
    sns.barplot(y=label_series.index, x=label_series)
    plt.savefig("flickr30k/label_counts.png")
    plt.title("Label frequency Flickr30k")
    plt.show()


    # get training splits
    test_ids = pd.read_csv("/raid/datasets/flickr30k_entities/test.txt", sep="\n", header=None)
    train_ids = pd.read_csv("/raid/datasets/flickr30k_entities/train.txt", sep="\n", header=None)
    val_ids = pd.read_csv("/raid/datasets/flickr30k_entities/val.txt", sep="\n", header=None)
    print("Num train samples: ", len(train_ids))
    print("Num val samples: ", len(val_ids))
    print("Num test samples: ", len(test_ids))
    
    #return img_paths, captions, labels, specific_inv_dict, broad_inv_dict, specific_dict, broad_dict, all_label_names
    return img_paths, img_ids, captions, labels, train_ids, val_ids, test_ids, label_names


def load_coco(path):
    img_folder = os.path.join(path, "images")
    imgs = [i for i in os.listdir(img_folder) if is_img(i)]
    imgs = sorted(imgs, key = lambda x: int(x.split(".")[0].split("_")[1]))
    img_ids = [int(x.split(".")[0]) for x in imgs]
    img_paths = [os.path.join(img_folder, i) for i in imgs]

    # load captions
    train_caps = json.load(open("annotations/captions_train2014.json", "r"))
    file_name = train_caps["images"][0]["file_name"]
    id_ = train_caps["images"][0]["id"]
    
    # load captions
    captions = pd.read_csv(os.path.join(path, "captions.txt"), delimiter="|")
    captions["img_idx"] = captions["image_name"].apply(lambda x: int(x.split(".")[0]))
    captions = captions.sort_values("img_idx")
    del captions["img_idx"]
    return img_paths, captions


    return img_paths, img_ids, captions, label_strs, train_ids, val_ids, test_ids, label_names


def load_dataset(dataset):
    paths = {"flickr30k": "/raid/datasets/f30k",
             "mscoco": "/raid/datasets/coco"}
    path = paths[dataset]

    # define dataset specifics
    if dataset == "flickr30k":
        label_path = "flickr30k/labels.npy"
        load_func = load_flickr
    elif dataset == "coco":
        label_path = "coco/labels.npy"
        load_func = load_coco

    # load or create
    if not os.path.exists(label_path):
        out = load_func(path)
        np.save(label_path, out)
    else:
        out = np.load(label_path, allow_pickle=True)
    img_paths, img_ids, captions, label_strs, train_ids, val_ids, test_ids, label_names = out

    # check correctness of lens
    assert len(img_paths) == len(img_ids)
    assert len(img_paths) == len(train_ids) + len(val_ids) + len(test_ids)
    assert len(captions) % len(img_ids) == 0
    assert len(img_ids) == len(label_strs)
    return out
