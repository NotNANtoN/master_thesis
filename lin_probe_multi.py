import os

import json
import pandas as pd
import hydra



def linear_probe(cfg, feats, label_names, steps_per_epoch, pos_fraction, folder_name):
    [train_img_features, train_labels], [val_img_features, val_labels], [test_img_features, test_labels] = feats

    import torch
    # calculate and log linear probe auc using logistic regression
    from torch.utils.data import TensorDataset
    train_ds = TensorDataset(train_img_features, train_labels)
    val_ds = TensorDataset(val_img_features, val_labels)
    test_ds = TensorDataset(test_img_features, test_labels)
    def make_dl(ds, train=True):
        return torch.utils.data.DataLoader(ds, batch_size=512, shuffle=train, pin_memory=True, num_workers=4)
    train_dl = make_dl(train_ds, train=True)
    val_dl = make_dl(val_ds, train=False)
    test_dl = make_dl(test_ds, train=False)

    
    feat_size = train_img_features.shape[1]
    
    # create model    
    model = torch.nn.Linear(feat_size, len(label_names))
    
    from supervised_models import LitCLIP
    lit_model = LitCLIP(model, label_names,
                        cfg["max_epochs"], cfg["lr"], steps_per_epoch, 
                        weight_decay=cfg["weight_decay"], use_pos_weight=cfg["use_pos_weight"],
                        pos_fraction=pos_fraction, eps=cfg["eps"], beta1=cfg["beta1"], beta2=cfg["beta2"])
    
        
    from learning_utils import init_trainer#, init_test_dms
    #cfg["val_check_interval"] = min(int(cfg["val_check_interval"] * (32 / cfg["batch_size"])), len(data_module.train_dataloader()))
    cfg["val_check_interval"] = 0.9
    root_folder = "/raid/8wiehe/"
    trainer = init_trainer(root_folder, cfg, "linear_probe", num_sanity_val_steps=0)
    # fit model
    trainer.fit(lit_model, train_dl, val_dl)
        
    val_preds, test_preds = trainer.predict(lit_model, [val_dl, test_dl])
    
    
    val_preds = torch.cat(val_preds)
    test_preds = torch.cat(test_preds)

    # eval model
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import average_precision_score
    import numpy as np
    import json
    
    save_folder = "results/linear_probe"
    
    import datetime
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    folder_name = f"{datetime_str}_{folder_name}"
    os.makedirs(os.path.join(save_folder, folder_name), exist_ok=True)
    
    rows = []
    for labels, preds, name in (
        (val_labels, val_preds, "val"),
        (test_labels, test_preds, "test")
    ):
        labels = labels.numpy()
        preds = preds.cpu().numpy()
        
        aucs = roc_auc_score(labels, preds, average=None)
        aps = average_precision_score(labels, preds, average=None)
        
        auc_dict = {label_names[i]: aucs[i] for i in range(len(label_names))}
        auc_dict["macro"] = np.mean(aucs)
        auc_dict["micro"] = roc_auc_score(labels, preds, average="micro")
        auc_dict["metric"] = "auc"
        
        ap_dict = {label_names[i]: aps[i] for i in range(len(label_names))}
        ap_dict["macro"] = np.mean(aps)
        ap_dict["micro"] = average_precision_score(labels, preds, average="micro")
        ap_dict["metric"] = "ap"
        
        row = pd.DataFrame([auc_dict, ap_dict])
        row["split"] = name
        rows.append(row)
        
        with open(os.path.join(save_folder, folder_name, f"{name}_auc.json"), "w") as f:
            json.dump(auc_dict, f)
        with open(os.path.join(save_folder, folder_name, f"{name}_ap.json"), "w") as f:
            json.dump(ap_dict, f)
        
    # save cfg
    with open(os.path.join(save_folder, folder_name, "cfg.json"), "w") as f:
        json.dump(cfg, f)
            

    
    import wandb
    wandb.finish()
    
    df = pd.concat(rows)
    
    return folder_name, df

@hydra.main(config_path="conf", config_name="supervised")
def main(cfg):    
    # switch back to original wd using hydra
    os.chdir(hydra.utils.get_original_cwd())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    from linear_probe_utils import setup_linear_probe
    out = setup_linear_probe(cfg)
    cfg, feats, label_names, steps_per_epoch, pos_fraction, folder_name = out
    
    
    seeds = [cfg["seed"]] if cfg["num_seeds"] <= 1 else [cfg["seed"] + i for i in range(cfg["num_seeds"])]

    df_list = []
    for seed in seeds:
        import pytorch_lightning
        pytorch_lightning.seed_everything(seed)
        
        folder_name_seed, df = linear_probe(cfg, feats, label_names, steps_per_epoch, pos_fraction, folder_name)
        
        df["seed"] = seed
        df_list.append(df)


        
    if len(seeds) > 1:
        # save aggregated results
        save_folder = "results/linear_probe_agg"
        os.makedirs(os.path.join(save_folder, folder_name_seed), exist_ok=True)
        
        agg_df = pd.concat(df_list)
        agg_df.groupby("metric").apply(lambda met_df: met_df.groupby("split").mean()).reset_index().to_csv(os.path.join(save_folder, folder_name_seed, "means.csv"))
        agg_df.groupby("metric").apply(lambda met_df: met_df.groupby("split").std()).reset_index().to_csv(os.path.join(save_folder, folder_name_seed, "stds.csv"))
        
        # save cfg
        with open(os.path.join(save_folder, folder_name_seed, "cfg.json"), "w") as f:
            json.dump(cfg, f)
       

    
if __name__ == "__main__":
    main()