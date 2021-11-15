import os
import torch
import clip
from tqdm.auto import tqdm
from PIL import Image


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        super().__init__()
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]))


@torch.inference_mode()
def get_image_features(img_paths, model, transform, device, load_path, batch_size=16, save=True):
    if save and os.path.exists(load_path):
        all_feats = torch.load(load_path)
    else:
        
        ds = ImgDataset(img_paths, transform)
        dl = torch.utils.data.DataLoader(ds, 
                                         batch_size=batch_size, 
                                         pin_memory=False,
                                         num_workers=16)
        
        all_feats = torch.zeros(len(img_paths), 512)
        
        for i, img_batch in enumerate(tqdm(dl)):
            feats = model.encode_image(img_batch.to(device)).cpu()
            all_feats[i * batch_size: i * batch_size + batch_size] = feats
        if save:
            torch.save(all_feats, load_path)
    return all_feats


@torch.inference_mode()
def get_text_features(texts, model, device, load_path, batch_size=16, save=True):
    if save and os.path.exists(load_path):
        all_feats = torch.load(load_path)
    else:
        all_feats = torch.zeros(len(texts), 512)
        for i in tqdm(list(range(0, len(texts), batch_size))):
            text_batch = texts[i: i + batch_size]
            tokenized = clip.tokenize(text_batch, truncate=True).to(device)
            text_feats = model.encode_text(tokenized)
            text_feats = text_feats.cpu().float()
            all_feats[i: i + batch_size] = text_feats
        if save:
            torch.save(all_feats, load_path)
    return all_feats
