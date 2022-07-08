import os

import torch
import pytorch_lightning


@torch.inference_mode()
def get_preds(model, dl):
    #model.eval()
    preds = []
    targets = []
    for batch in dl:
        x, y = batch[0], batch[1]
        preds.append(model(x.cuda()).detach())
        targets.append(y)
    preds = torch.cat(preds, dim=0).cpu().numpy()
    targets = torch.cat(targets, dim=0).numpy()
    return preds, targets

    
def calc_metrics(preds, targets):
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
    ap = average_precision_score(targets, preds)
    roc_auc = roc_auc_score(targets, preds)
    #pr_auc = auc(precision_recall_curve(targets, preds)[0], precision_recall_curve(targets, preds)[1])
    metrics = {"ap": [ap], "roc_auc": [roc_auc]}#, "pr_auc": pr_auc}
    import pandas as pd
    return pd.DataFrame(metrics)
    
def eval_dl(model, dl, path):
    preds, targets = get_preds(model, dl)
    metrics = calc_metrics(preds, targets)
    # store metrics at path
    metrics.to_csv(path + "_metrics.csv", index=False)
    return metrics

def eval_supervised_model(model, dm: pytorch_lightning.LightningDataModule, save_path):
    """ 
    Calculate essential metrics like AUC, AP for micro and macro on validation and test set
    """
    model.cuda()
    # val loader
    val_dl = dm.val_dataloader()
    val_metrics = eval_dl(model, val_dl, os.path.join(save_path, "val"))
    # test loader
    test_dl = dm.test_dataloader()
    test_metrics = eval_dl(model, test_dl, os.path.join(save_path, "test"))
    model.cpu()
    return val_metrics, test_metrics
