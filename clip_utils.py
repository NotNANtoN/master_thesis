import os
from regex import S

import torch
import clip
from tqdm.auto import tqdm
from PIL import Image
import pytorch_lightning
import torchvision.transforms as TF
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


from dataset_utils import load_dataset


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform):
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]
    

class SubsampleSents(torch.nn.Module):
    def __init__(self, frac, randomize_order, min_sent_len=5):
        super().__init__()
        self.frac = frac
        self.min_sent_len = min_sent_len
        self.randomize_order = randomize_order
        
    def forward(self, x):
        sents = [s.replace("_", "") for s in x.split(".")]
        sents = [s.strip() for s in sents]
        sents = [s for s in sents if len(s) >= self.min_sent_len]
        
        if self.randomize_order:
            np.random.shuffle(sents)
        if self.frac < 1.0:
            num_sents_used = round(len(sents) * self.frac)
            idcs = list(range(len(sents)))
            idcs_used = sorted(np.random.choice(idcs, num_sents_used, replace=False))
            sents = [sents[i] for i in idcs_used]
        text = ". ".join(sents)
        return text


class CLDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, captions, transform, 
                 sent_frac=1.0, randomize_order=False):
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.captions = captions
        self.transform = transform
        self.sent_frac = sent_frac
        self.randomize_order = randomize_order
        
        self.sent_subsampler = SubsampleSents(self.sent_frac, randomize_order)
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        img = self.transform(Image.open(self.paths[i]).convert("RGB"))
        labels = self.labels[i]
        text = self.captions[i]
        
        # add data agumentations for text
        if self.sent_frac < 1.0 or self.randomize_order:
            text = self.sent_subsampler(text)
            
        # tokenize text
        tokenized_text = clip.tokenize(text, truncate=True).squeeze()
        
        return img, labels, tokenized_text
    

class FinetuneDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, model, transform,
                 dataset_name: str = "mimic-cxr",
                 batch_size: int = 32,
                 mode="full",
                 augs=None,
                 sent_frac=0.8,
                 use_cl=False,
                 root_folder="",
                 num_workers=4,
                 dataset_size=None,
                 pin_memory=False,
                 randomize_order=False,
                 seed=42,
                ):
        super().__init__()
        self.dataset_name = dataset_name
        self.bs = batch_size
        self.mode = mode
        self.root_folder = root_folder
        self.transform = transform
        self.sent_frac = sent_frac
        self.num_workers = num_workers
        self.dataset_size = dataset_size
        self.pin_memory = pin_memory
        self.augs = augs
        
        # load dataset
        df, label_names = load_dataset(dataset_name)
        self.label_names = label_names
        self.num_labels = len(label_names)
        # create masks
        train_mask = (df["split"] == "train").to_numpy()
        val_mask = ((df["split"] == "validate") | (df["split"] == "val")).to_numpy()
        test_mask = (df["split"] == "test").to_numpy()
        
        if self.dataset_size is not None:
            train_count = train_mask.sum()
            remove_count = int((1 - self.dataset_size) * train_count)
            np.random.seed(seed)
            # set train_reduced_count random parts of train_mask, where train_mask is currently True, to False
            train_mask[np.random.choice(np.where(train_mask)[0], remove_count, replace=False)] = False
        elif 1 == 0:
            # apply dataset_size to train mask by choosing appropriate distribution depending on labels columns
            train_labels = np.stack(df["labels"][train_mask].to_numpy())
            support_idcs = []
            # for each label, select the first dataset_size samples where that label is true
            for i in range(len(label_names)):
                feat_labels = train_labels[:, i]
                # get masks of pos/neg labels
                true_mask = feat_labels == 1
                # do not reuse indices we added before
                true_mask[support_idcs] = 0
                # get idcs
                feat_supp_idcs = np.where(true_mask)[0][:self.dataset_size]            
                # add to support idcs
                support_idcs.extend(feat_supp_idcs.tolist())
                
            
            filter_minimum_inclusive_set = False
            if filter_minimum_inclusive_set:
                # now we have the indices that include each label at least dataset_size times
                # but we will throw out samples that do not belong to the minimum inclusive set (i.e. smallest set where at least dataset_size labels are true per feature)
                train_labels[train_labels != 1] = 0  # set to 0 to sum across axis
                train_labels = train_labels.astype(int)
                
                import numba
                @numba.jit()
                def should_remove(idx, support_idcs, train_labels, dataset_size):
                    all_but_one = np.array([i for i in support_idcs if i != idx])
                    # labels without that idx
                    remaining_labels = train_labels[all_but_one]
                    # check if the count of each label per feature is at least dataset_size, if that is the case remove the idx from support idcs
                    return not (sum(np.sum(remaining_labels, axis=0) < dataset_size) > 0)
                for idx in tqdm(support_idcs[:]):
                    if should_remove(idx, np.array(support_idcs), train_labels, dataset_size):
                        support_idcs.remove(idx)
                        
                print("Num idcs after removal: ", len(support_idcs))
            # create new train mask where only support idcs (minimum inclusive set) are true
            # which of the original train idcs to we keep?
            train_idcs = np.arange(len(train_mask))[train_mask]
            keep_idcs = train_idcs[support_idcs]
            # overwrite train mask
            train_mask = np.array([i in keep_idcs for i in range(len(df))])                
                
        # apply splits
        train_labels = np.stack(df["labels"][train_mask].to_numpy())
        val_labels = np.stack(df["labels"][val_mask].to_numpy())
        test_labels = np.stack(df["labels"][test_mask].to_numpy())
        # set labels to 0
        train_labels[np.isnan(train_labels)] = 0
        val_labels[np.isnan(val_labels)] = 0
        test_labels[np.isnan(test_labels)] = 0
        train_labels[train_labels == -1] = 0
        val_labels[val_labels == -1] = 0
        test_labels[test_labels == -1] = 0
        # to float
        train_labels = train_labels.astype(float)
        val_labels = val_labels.astype(float)
        test_labels = test_labels.astype(float)

        if mode == "freeze":
            # load feats
            img_features, caption_features = get_clip_img_caption_features(df, model, transform, dataset_name, bs=128)
            # get feats
            train_clip_feats = img_features[train_mask]
            val_clip_feats = img_features[val_mask]
            test_clip_feats = img_features[test_mask]
            # create feature tensor datasets
            self.train_ds = TensorDataset(train_clip_feats, train_labels)
            self.val_ds = TensorDataset(val_clip_feats, train_labels)
            self.test_ds = TensorDataset(test_clip_feats, train_labels)
            self.bs = 512
        else:
            if self.augs is None:
                print("No training set augmentation specified!")
                train_transform = transform
            else:
                train_transform = self.augs
            
            # get paths 
            train_paths = df["img_path"][train_mask].to_list()
            val_paths = df["img_path"][val_mask].to_list()
            test_paths = df["img_path"][test_mask].to_list()
            # get texts (only used in CL mode)
            if "caption" not in df.columns:
                df["caption"] = np.nan
            train_texts = df["caption"][train_mask].to_list()
            val_texts = df["caption"][val_mask].to_list()
            test_texts = df["caption"][test_mask].to_list()
                
            if use_cl:
                # create ds                
                self.train_ds = CLDataset(train_paths, train_labels, train_texts, train_transform, 
                                          sent_frac=sent_frac, randomize_order=randomize_order,)
                self.val_ds = CLDataset(val_paths, val_labels, val_texts, transform)
                self.test_ds = CLDataset(test_paths, test_labels, test_texts, transform)
            else:
                # create image datasets
                self.train_ds = ImageDataset(train_paths, train_labels, train_transform)
                self.val_ds = ImageDataset(val_paths, val_labels, transform)
                self.test_ds = ImageDataset(test_paths, test_labels, transform)
            
        self.steps_per_epoch = len(self.train_ds) // self.bs + (len(self.train_ds) % self.bs != 0)
        self.pos_fraction = torch.from_numpy(train_labels.mean(axis=0))

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, num_workers=self.num_workers, shuffle=True,
                              batch_size=self.bs,# persistent_workers=1,
                              pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_ds, num_workers=self.num_workers,
                              shuffle=False, batch_size=128, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_ds, num_workers=self.num_workers,
                              shuffle=False, batch_size=128, pin_memory=self.pin_memory)



class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        super().__init__()
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]))

    
def apply_train_mode(mode, model):
    if "RN" in model.name:
        for p in model.parameters():
            p.requires_grad = True
    # freeze certain layers or add adapters
    elif mode == "freeze":
        for p in model.parameters():
            p.requires_grad = False
    elif mode == "train_norm":
        for n, p in model.named_parameters():
            p.requires_grad = ".ln_" in n or ".bn_" in n
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "train_mlp_norm":
        for n, p in model.named_parameters():
            p.requires_grad = ".ln_" in n or ".bn_" in n or ".mlp." in n
    elif mode == "adapters":
        #from contrastive_learning_utils import LitCLCLIP
        #from supervised_models import CLIPClassifier
        from clip.model import VisionTransformer
        if isinstance(model, clip.model.CLIP) or isinstance(model, VisionTransformer):# or isinstance(model, LitCLCLIP)# or isinstance(model, CLIPClassifier):
            for n, p in model.named_parameters():
                p.requires_grad = ".ln_" in n or "adapter" in n
        else:
            raise NotImplementedError(f"Adapter training not supported for {model.name} of type {type(model)}")
    elif mode == "adapters_correct":
        if isinstance(model, clip.model.CLIP) or isinstance(model, VisionTransformer):# or isinstance(model, LitCLCLIP)# or isinstance(model, CLIPClassifier):
            for n, p in model.named_parameters():
                p.requires_grad = "adapter" in n
        else:
            raise NotImplementedError(f"Adapter training not supported for {model.name} of type {type(model)}")
    else:
        raise NotImplementedError(f"Unknown training mode {mode}")

    
def load_clip(clip_name, mode, device=None, down_sample_size=16, adapter_flow="hard"):
    # clip_models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = clip.load(clip_name, jit=False, device=device)
    model.device = device
    clip_load_name = clip_name.replace("/", "_")
    model.name = clip_load_name
    
    if "RN" not in clip_name:
        if "adapters" in mode:
            from adapter_utils import add_adapters
            model = add_adapters(model, down_sample_size, adapter_flow=adapter_flow)
        
    apply_train_mode(mode, model)
    return model, transform, clip_load_name


def get_clip_img_caption_features(df, model, transform, dataset_name, bs=32):
    # create folder
    load_path = os.path.join(dataset_name, "feats")
    os.makedirs(load_path, exist_ok=True)
    # get features
    no_model_given = model is not None and not isinstance(model, str)
    device = "cuda" # model.device if no_model_given else None
    model_name = model.name if no_model_given else model
    img_features = get_image_features(df["img_path"].to_list(), model, transform, device, 
                                      os.path.join(load_path, f"{model_name}_img_feats.pt"), 
                                      batch_size=bs, save=True)
    caption_features = get_text_features(df["caption"], model, device, 
                                         os.path.join(load_path, f"{model_name}_caption_feats.pt"), 
                                         batch_size=bs, save=True)
    return img_features, caption_features


@torch.inference_mode()
def get_image_features(img_paths, model, transform, device, load_path, batch_size=16, save=True):
    if save and os.path.exists(load_path):
        return torch.load(load_path)
    
    ds = ImgDataset(img_paths, transform)
    dl = torch.utils.data.DataLoader(ds, 
                                        batch_size=batch_size, 
                                        pin_memory=False,
                                        num_workers=16)
    
   
    print(model)
    print(ds[0].shape)
    shape = model.encode_image(ds[0].unsqueeze(0).to(device)).cpu().shape[-1]
    all_feats = torch.zeros(len(img_paths), shape)
    
    for i, img_batch in enumerate(tqdm(dl)):
        feats = model.encode_image(img_batch.to(device)).cpu()
        all_feats[i * batch_size: i * batch_size + batch_size] = feats
    if save:
        torch.save(all_feats, load_path)
    return all_feats


@torch.inference_mode()
def get_text_features(texts, model, device, load_path, batch_size=16, save=True):
    if save and os.path.exists(load_path):
        return torch.load(load_path)
    shape = model.encode_text(clip.tokenize(texts[:2], truncate=True).to(device)).cpu().shape[-1]
    all_feats = torch.zeros(len(texts), shape)
    for i in tqdm(list(range(0, len(texts), batch_size))):
        text_batch = texts[i: i + batch_size]
        tokenized = clip.tokenize(text_batch, truncate=True).to(device)
        text_feats = model.encode_text(tokenized)
        text_feats = text_feats.cpu().float()
        all_feats[i: i + batch_size] = text_feats
    if save:
        torch.save(all_feats, load_path)
    return all_feats
