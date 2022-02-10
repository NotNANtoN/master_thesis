from typing import List

import torch
import numpy as np
from tqdm.auto import tqdm

from clip_utils import get_image_features, get_text_features


def calc_metrics(feats, caption_features):
    sims = torch.cosine_similarity(feats, caption_features)
    #pred_idcs: List[int] = sims.topk(k=10).indices.tolist()
    pred_idcs = sims.topk(k=10).indices
    return pred_idcs


def calc_img2text_retrieval(img_features, caption_features, cap_per_img=5):
    #calc_metrics = torch.jit.trace(calc_metrics, (img_features[0].unsqueeze(0), caption_features))
    r1 = torch.zeros(img_features.shape[0])
    r5 = torch.zeros(img_features.shape[0])
    r10 = torch.zeros(img_features.shape[0])
    for img_idx in tqdm(torch.arange(img_features.shape[0], device=img_features.device)):
        feats = img_features[img_idx].unsqueeze(0)
        #print(feats.device, caption_features.device, img_idx.device)
        pred_idcs = calc_metrics(feats, caption_features)
        ground_truth_idcs = torch.arange(cap_per_img, device=feats.device) + img_idx * cap_per_img
        r1_val = any([pred_idcs[i] in ground_truth_idcs for i in range(1)])
        r5_val = any([pred_idcs[i] in ground_truth_idcs for i in range(5)])
        r10_val = any([pred_idcs[i] in ground_truth_idcs for i in range(10)])
        r1[img_idx] = r1_val
        r5[img_idx] = r5_val
        r10[img_idx] = r10_val
    return r1, r5, r10


def calc_text2img_retrieval(caption_features, img_features, cap_per_img=5):
    r1 = torch.zeros(caption_features.shape[0])
    r5 = torch.zeros(caption_features.shape[0])
    r10 = torch.zeros(caption_features.shape[0])
    for cap_idx in tqdm(torch.arange(caption_features.shape[0], device=caption_features.device)):
        feats = caption_features[cap_idx].unsqueeze(0)
        #print(feats.device, caption_features.device, img_idx.device)
        pred_idcs = calc_metrics(feats, img_features)
        ground_truth_idx = cap_idx // cap_per_img
        r1_val = (pred_idcs[:1] == ground_truth_idx).max()
        r5_val = (pred_idcs[:5] == ground_truth_idx).max()
        r10_val = (pred_idcs[:10] == ground_truth_idx).max()
        r1[cap_idx] = r1_val
        r5[cap_idx] = r5_val
        r10[cap_idx] = r10_val
    return r1, r5, r10


def add_prompt(labels, prefix="", suffix=""):
    new_labels = [prefix + l + suffix for l in labels]
    return new_labels


def norm(a):
    return a / a.norm(dim=-1, keepdim=True)


def label_strs_to_int(label_strs, label_names):
    return torch.tensor([l in label_strs for l in label_names]).long()


def create_label_encs(model, label_names, prefixes, use_multi_label_setting=False, use_norm=False):
    device = next(model.parameters()).device
    if use_multi_label_setting:
        # create queries
        label_queries = [add_prompt(label_names, prefix=prefix) for prefix in prefixes]
        # encode queries
        label_encs = [get_text_features(label_query, model, device, 
                                             None,#os.path.join(load_path, f"{clip_name}_caption_feats.pt"), 
                                             batch_size=32, save=False) for label_query in label_queries]
        label_encs = torch.stack(label_encs).mean(dim=0)
        print(label_encs.shape)
    else:
        # create baseline query, then positive query and construct negative encoding by 
        # subtracting pos encoding minus baseline encoding
        all_diff_encs = []
        for prefix in prefixes:
            baseline_enc = get_text_features([prefix], model, device, None, batch_size=1, save=False)[0]
            pos_encs = get_text_features(add_prompt(label_names, prefix=prefix), model, device, None, batch_size=32, save=False)
            if use_norm:
                diff_encs = pos_encs - baseline_enc#norm(norm(pos_encs) - norm(baseline_enc))
            else:
                diff_encs = pos_encs - baseline_enc
            all_diff_encs.append(diff_encs)
        diff_enc = norm(torch.stack(all_diff_encs)).mean(dim=0)
        if use_norm:
            neg_encs = norm(baseline_enc - diff_enc)#norm(norm(baseline_enc) - norm(diff_enc))
        else:
            neg_encs = baseline_enc - diff_enc

        #label_encs = torch.stack(neg_encs)
        label_encs = [pos_encs, neg_encs]
    return label_encs


def calc_accuracies(img_features, label_encs):
    r1_sum = 0
    r5_sum = 0
    r10_sum = 0

    for img_idx in tqdm(range(len(img_paths))):
        img_enc = img_features[img_idx].unsqueeze(0)    
        sims = torch.cosine_similarity(img_enc, label_encs)
        pred_idcs = sims.topk(k=10).indices
        top_idx = pred_idcs[0]
        gt_labels = label_strs[img_idx]
        gt_label_idcs = torch.where(label_strs_to_int(gt_labels, label_names))[0]
        
        r1 = 0
        r5 = 0
        r10 = 0
        for i, pred in enumerate(pred_idcs):
            if pred in gt_label_idcs:
                if i == 0:
                    r1, r5, r10 = 1, 1, 1
                elif i < 5:
                    r5, r10 = 1, 1
                else:
                    r10 = 1
                break
        r1_sum += r1
        r5_sum += r5
        r10_sum += r10
    return (r1_sum / (img_idx + 1), r5_sum / (img_idx + 1), r10_sum / (img_idx + 1))


def calc_binary_acc(img_features, labels, label_names, pos_label_encs, neg_label_encs):
    # make list if not list yet
    if not isinstance(pos_label_encs, list):
        pos_label_encs = [pos_label_encs]
        neg_label_encs = [neg_label_encs]
    
    accs = []
    for img_idx in tqdm(range(len(img_features))):
        gt_labels = labels[img_idx]
        gt_label_idcs = torch.where(label_strs_to_int(gt_labels, label_names))[0]
        gt_binary = torch.zeros(len(label_names)).bool()
        gt_binary[gt_label_idcs] = True
        
        img_enc = img_features[img_idx].unsqueeze(0)    
        
        img_pos_sims = []
        img_neg_sims = []
        for pos_encs, neg_encs in zip(pos_label_encs, neg_label_encs):
            pos_sims = torch.cosine_similarity(img_enc, pos_encs)
            neg_sims = torch.cosine_similarity(img_enc, neg_encs)
            img_pos_sims.append(pos_sims)
            img_neg_sims.append(neg_sims)
            
            
        pos_sims = torch.stack(img_pos_sims).mean(dim=0)
        neg_sims = torch.stack(img_neg_sims).mean(dim=0)

        binary_preds = pos_sims > neg_sims
        #binary_preds = torch.zeros(len(label_names))
        preds = torch.where(binary_preds)[0]
        acc = (binary_preds == gt_binary).float().mean()
        #img_accs.append(acc)
        #acc = torch.mean(torch.stack(img_accs))
        #print(gt_label_idcs)
        #print(len(preds))
        #print(acc)
        #print()
        accs.append(acc)

    return accs