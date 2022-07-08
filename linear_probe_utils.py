import os

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


@torch.inference_mode()
def get_feats(model, dl, use_tqdm=False, device="cuda", name=None):
    path = os.path.exists(os.path.join("features", name))
    if name is not None and path:
        return torch.load(os.path.join("features", name))
    # get image features and labels
    img_features = []
    labels = []
    for batch in tqdm(dl, disable=not use_tqdm):
        imgs, batch_labels = batch
        if hasattr(model, "encode_image"):
            feats = model.encode_image(imgs.to(device))
        else:
            feats = model(imgs.to(device))
        feats = F.normalize(feats, dim=-1).cpu()
        img_features.append(feats)
        labels.append(batch_labels.cpu())
    img_features = torch.cat(img_features)
    labels = torch.cat(labels)
    if name is not None:
        os.makedirs("features", exist_ok=True)
        torch.save((img_features, labels), os.path.join("features", name))
    return img_features, labels


def fit_and_eval_log_reg(train_feat_labels, val_feat_labels, test_feat_labels,
                             train_img_features, val_img_features, test_img_features,
                             label_name,
                             queue=None):
    
    
    # import and fit logistic regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(solver="saga", max_iter=200)
    lr.fit(train_img_features, train_feat_labels)
    
    # get validation and test predictions
    val_preds = lr.predict_proba(val_img_features)[:, 1]
    test_preds = lr.predict_proba(test_img_features)[:, 1]
    # calculate auc
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import average_precision_score
    
    if test_feat_labels.sum(axis=0) > 0:
        test_auc = roc_auc_score(test_feat_labels, test_preds)
        test_ap = average_precision_score(test_feat_labels, test_preds)
    else:
        test_auc, test_ap = None, None
    if val_feat_labels.sum(axis=0) > 0:
        val_auc = roc_auc_score(val_feat_labels, val_preds)
        val_ap = average_precision_score(val_feat_labels, val_preds)
    else:
        val_auc, val_ap = None, None
        
    if queue is not None:
        queue.put((label_name, val_auc, val_ap, test_auc, test_ap))
    return label_name, val_auc, val_ap, test_auc, test_ap


def binary_relevance_linear_probe(label_names, 
                                  train_img_features, train_labels ,
                                  val_img_features, val_labels,
                                  test_img_features, test_labels):
    
    from multiprocessing import Queue, Process
   
    q = Queue()
    
    pbar = tqdm(range(train_labels.shape[-1]))
    
    processes = []
    
    for i in pbar:
        label_name = label_names[i]
        
        pbar.set_description(f"{label_name}")
        pbar.update(1)
        
        # select 
        train_feat_labels = train_labels[:, i]
        test_feat_labels = test_labels[:, i]
        val_feat_labels = val_labels[:, i]
        
        # select feats where label is not nan
        train_img_features_no_nan = train_img_features[~np.isnan(train_feat_labels)]
        val_img_features_no_nan = val_img_features[~np.isnan(val_feat_labels)]
        test_img_features_no_nan = test_img_features[~np.isnan(test_feat_labels)]
        
        # remove nans
        train_feat_labels = train_feat_labels[~np.isnan(train_feat_labels)]
        test_feat_labels = test_feat_labels[~np.isnan(test_feat_labels)]
        val_feat_labels = val_feat_labels[~np.isnan(val_feat_labels)]
        
        # if train, val or test is empty or has only one label type, skip
        if len(train_feat_labels) == 0 or len(test_feat_labels) == 0 or len(val_feat_labels) == 0:
            continue
        elif len(np.unique(train_feat_labels)) == 1 or len(np.unique(test_feat_labels)) == 1 or len(np.unique(val_feat_labels)) == 1:
            continue
        
        
        #val_auc, val_ap, test_auc, test_ap = fit_and_eval_log_reg(train_feat_labels, val_feat_labels, test_feat_labels,
        #                                                            train_img_features, val_img_features, test_img_features)
        p = Process(target=fit_and_eval_log_reg, args=(train_feat_labels, val_feat_labels, test_feat_labels, 
                                                       train_img_features_no_nan, val_img_features_no_nan, test_img_features_no_nan,
                                                       label_name), 
                                                        kwargs={"queue": q})
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
        
    val_aucs = {}
    val_aps = {}
    test_aucs = {}
    test_aps = {}
    for p in processes:
        label_name, val_auc, val_ap, test_auc, test_ap = q.get()
        val_aucs[label_name] = val_auc
        val_aps[label_name] = val_ap
        test_aucs[label_name] = test_auc
        test_aps[label_name] = test_ap
    
    test_aucs["macro"] = sum(test_aucs.values()) / len(test_aucs)
    test_aps["macro"] = sum(test_aps.values()) / len(test_aps)
    val_aucs["macro"] = sum(val_aucs.values()) / len(val_aucs)
    val_aps["macro"] = sum(val_aps.values()) / len(val_aps)
    return test_aucs, test_aps, val_aucs, val_aps


def setup_linear_probe(cfg):
    cfg = dict(cfg)
    if cfg["batch_size"] is None:
        cfg["batch_size"] = 256
    
    # to fix pytorch issue with "Too many files open"
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    # seed everything using pytorch lightning
    import pytorch_lightning
    pytorch_lightning.seed_everything(cfg["seed"])
    
    
    # import torch-related stuff only after setting the gpu
    from linear_probe_utils import get_feats, binary_relevance_linear_probe
    from supervised_utils import init_model
    
    
    from clip_utils import FinetuneDataModule
    base_model, transform_basic, transform_aug, name = init_model(cfg)

    # extract real CLIP model
    if hasattr(base_model, "model"):
        base_model = base_model.model
    # if densenet, get features of its last conv layer
    if "densenet" in name:
        import torch
        import torch.nn.functional as F
        class DenseNetFeats(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                features = self.model(x)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                out = torch.flatten(out, 1)
                return out
                
        base_model = DenseNetFeats(base_model.features)
    
    base_model = base_model.cuda()
    base_model.eval()


    data_module = FinetuneDataModule(base_model, transform_basic, 
                                    dataset_name=cfg["dataset_name"], 
                                    mode=cfg["mode"], 
                                    batch_size=cfg["batch_size"],
                                    augs=transform_aug,
                                    use_cl=False)
    label_names = data_module.label_names
    train_dl, val_dl, test_dl = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
    
    folder_name = f"{name}_{cfg['dataset_name']}_{cfg['pretrained']}"

    
    train_feat_name =  folder_name + "_train_feats.pt"
    val_feat_name = folder_name + "_val_feats.pt"
    test_feat_name = folder_name + "_test_feats.pt"
    train_img_features, train_labels = get_feats(base_model, train_dl, use_tqdm=True, name=train_feat_name)
    val_img_features, val_labels = get_feats(base_model, val_dl, name=val_feat_name)
    test_img_features, test_labels = get_feats(base_model, test_dl, name=test_feat_name)
    feats = [train_img_features, train_labels], [val_img_features, val_labels], [test_img_features, test_labels]
    return cfg,feats, label_names, data_module.steps_per_epoch, data_module.pos_fraction, folder_name


