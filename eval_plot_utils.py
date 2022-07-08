import os
import json

import pandas as pd


def load_sl_results():
    folder = "results/supervised"
    files = os.listdir(folder)
    paths = [os.path.join(folder, f) for f in files]

    # filter if only cfg.json is in directory at path    
    paths =  [p for p in paths if len(os.listdir(p)) > 2]

    # read cfgs, test_metrics and val metrics
    rows = []
    for p in paths:
        try: 
            cfg = json.load(open(os.path.join(p, "cfg.json")))
            val_metrics = pd.read_csv(os.path.join(p, "val_metrics.csv"))
            test_metrics = pd.read_csv(os.path.join(p, "test_metrics.csv"))
        except FileNotFoundError:
            print(os.listdir(p))
            continue
        # prepend val_ and _test_ to column names
        val_metrics.columns = ["val_" + c for c in val_metrics.columns]
        test_metrics.columns = ["test_" + c for c in test_metrics.columns]
        # merge
        metrics = pd.concat([val_metrics, test_metrics, pd.DataFrame(cfg, index=[0])], axis=1)
        rows.append(metrics)
    df = pd.concat(rows, axis=0)

    # fill pretrained NA with 1
    df["pretrained"] = df["pretrained"].fillna(1)
    df["adapter_flow"] = df["adapter_flow"].fillna("hard") 
    # drop columns that only have one unique value
    df = df.loc[:, df.nunique() > 1]

    
    return df
    
    
def load_cl_results():
    folder = "results/cl"
    files = os.listdir(folder)
    paths = [os.path.join(folder, f) for f in files]

    # filter if only cfg.json is in directory at path    
    #paths =  [p for p in paths if len(os.listdir(p)) > 2]
    # read cfgs, test_metrics and val metrics
    
    print("Num paths found: ", len(paths))
    rows = []
    for p in paths:
        from json import JSONDecodeError
        try: 
            cfg = json.load(open(os.path.join(p, "cfg.json")))
        except FileNotFoundError:
            print(os.listdir(p))
            continue
        except JSONDecodeError:
            print("JSONDecodeError")
            continue
        
        # try opening results
        ext_dfs = []
        
        ext_names = ["covidx", "rsna", "chexpert"]
        for name in ext_names:
            ext_path = os.path.join(p, f"lin_probe_results_{name}.csv")
            if os.path.exists(ext_path):
                metrics = pd.read_csv(ext_path)
                metrics.columns = [name + "_" + c for c in metrics.columns]
                ext_dfs.append(metrics)
        if os.path.exists(os.path.join(p, "lin_probe_results.csv")):
            lin_probe_metrics = pd.read_csv(os.path.join(p, "lin_probe_results.csv"))
            lin_probe_metrics.columns = ["lin_probe_" + c for c in lin_probe_metrics.columns]
            ext_dfs.append(lin_probe_metrics)
        # zero shot results
        zero_shot_path = os.path.join(p, "zero_shot_aucs_.csv")
        if os.path.exists(zero_shot_path):
            metrics = pd.read_csv(zero_shot_path)
            ext_dfs.append(metrics)
        # try opening fine-tuning results
        fine_tune_results = []
        names = ["sl_adapters", "sl_full", "sl_new_adapters"]
        for name in names:
            if os.path.exists(os.path.join(p, name)):
                try:
                    test_auc = pd.read_csv(os.path.join(p, name, "test_metrics.csv"))["roc_auc"].iloc[0]
                    val_auc = pd.read_csv(os.path.join(p, name, "val_metrics.csv"))["roc_auc"].iloc[0]
                except FileNotFoundError:
                    continue
                clean_name = "_".join(name.split("_")[1:])
                fine_tune_df = pd.DataFrame({f"{clean_name}_test_auc": [test_auc],
                                                f"{clean_name}_val_auc": [val_auc]})
                fine_tune_results.append(fine_tune_df)
                                            
        # merge
        dfs = ext_dfs + [pd.DataFrame(cfg, index=[0])] + fine_tune_results

        metrics = pd.concat(dfs, axis=1)
        rows.append(metrics)
    df = pd.concat(rows, axis=0)

    print(df["adjust_grad_acc_to"].isna().mean())

    # fill pretrained NA with 1
    df["pretrained"] = df["pretrained"].fillna(1)
    df["adapter_flow"] = df["adapter_flow"].fillna("hard") 
    df["fixed_ds_subsampling_seed"] = df["fixed_ds_subsampling_seed"].fillna(0)
    df["adjust_grad_acc_to"] = df["adjust_grad_acc_to"].fillna(0)
    df["sl_dataset_size"] = df["sl_dataset_size"].fillna(1.0)
    if "fixed_val_ds_order" in df.columns:
        df["fixed_val_ds_order"] = df["fixed_val_ds_order"].fillna(0)
    if "missing_mode" in df.columns:
        df["missing_mode"] = df["missing_mode"].fillna("zeros")
    if "cyclic_lambda" in df.columns:
        df["cyclic_lambda"] = df["cyclic_lambda"].fillna(0)

    
    # drop columns that only have one unique value
    df = df.loc[:, df.nunique() > 1]
    
    exclude = ["full_", "adapters_", "check_interval", "ds_order"]
    # exchange val and test columns for rows where fixed_val_ds_order is 0
    val_col_names = [c for c in df.columns if "val" in c and not any(ex in c for ex in exclude)]
    test_col_names = [c for c in df.columns if "test" in c and not any(ex in c for ex in exclude)]
    mask = df["fixed_val_ds_order"] == 0
    temp_val_cols = df.loc[mask, val_col_names].copy()
    df.loc[mask, val_col_names] = df.loc[mask, test_col_names].to_numpy()
    df.loc[mask, test_col_names] = temp_val_cols.to_numpy()
    df = df.drop(columns=["fixed_val_ds_order"])
    
    
    # filter out test runs
    df = df[df["sl_max_epochs"].isna() | (df["sl_max_epochs"] == 10)]

    unnecessary_cols = ["do_sl", "do_ext", "gpu", "val_check_interval"]
    df = df.drop(columns=unnecessary_cols)

    return df



def cl_plot_ds_size(df, ax=None, mode="test", max_epochs=None):
    # make plot showing dataset size effect
    import matplotlib.pyplot as plt

    fixed_seed_issue = 1

    if ax is None:
        ax = plt.figure().gca()
        
    name = "ViT-B/32"
    
    if name == "ViT-B/32":
        lr = 3e-4
    else:
        lr = 3e-5
        
        
    model_df = df.copy()
    if "model_name" in model_df.columns:
        model_df = model_df[model_df["model_name"] == name]
    model_df = model_df[model_df["cyclic_lambda"] == 0]
    model_df = model_df[model_df["mode"] == "adapters"]
    model_df = model_df[model_df["randomize_order"] == 1]
    model_df = model_df[model_df["sent_frac"] == 1.0]
    model_df = model_df[model_df["lr"] == lr]
    model_df = model_df[model_df["batch_size"] == 192]
    model_df = model_df[model_df["adapter_flow"] == "easy"]
    model_df = model_df[model_df["num_gpus"] == 1]
    model_df = model_df[model_df["mixup_alpha"] == 0]
    model_df = model_df[model_df["sl_dataset_size"] == 1]
    model_df = model_df[model_df["dataset_size"] >= 0.01]
    

    model_df = model_df[model_df["fixed_ds_subsampling_seed"] == fixed_seed_issue]
    
    model_df = model_df.loc[:, model_df.nunique() > 1]
    print(model_df)    
    
    metrics = ["adapters_test_auc", "lin_probe_mean_test_auc"]#, "full_test_auc"]#, "new_adapters_test_auc"]
    labels = ["CL + SL", "CL + Linear Probe"]#, "Full"]#, "New adapters"]
    
    
    if mode == "val":
        metrics = [l.replace("test", "val") for l in metrics]
    
    max_epoch_vals = [10]
    if max_epochs is not None:
        max_epoch_vals += max_epochs
    
    if len(max_epoch_vals) > 1:
        all_labels = []
        for epoch_val in max_epoch_vals:
            epoch_labels = [f"{l} {epoch_val} Epochs" for l in labels]
            all_labels.extend(epoch_labels)
        labels = all_labels
        
    lin_probe_col = None
    count = 0
    
    for max_epoch in max_epoch_vals:
        for metric in metrics:
            label = labels[count]
            count += 1
            
            if max_epoch > 10 and "lin_probe" not in metric:
                continue
            
            print()
            print(metric)
        
                
            skip_msg = ""
            if len(model_df) == 0:
                print(skip_msg)
                continue
            #if df.nunique()["dataset_size"] < 2:
            #    print(skip_msg)
            #    continue
            if len(model_df.columns) == 0:
                print(skip_msg)
                continue

            epoch_df = model_df[model_df["max_epochs"] == max_epoch]
            epoch_df = epoch_df.sort_values("dataset_size")
            # average over same seed for same dataset size
            epoch_df = epoch_df.groupby("dataset_size").apply(lambda x: x.groupby("seed").mean().reset_index()).reset_index(drop=True)
            
            # get std over same dataset size
            std = epoch_df.groupby("dataset_size").std()[metric]
            std = std.fillna(0)
            print("STD: ", std)
            # take average over same dataset size
            means = epoch_df.groupby("dataset_size").mean().reset_index()
            
            style = "-X"
            if "lin_probe" in metric:
                if lin_probe_col is not None:
                    color = lin_probe_col
                else:
                    color = next(ax._get_lines.prop_cycler)["color"]
                    # if color is red, skip it 
                    if color == "r":
                        color = next(ax._get_lines.prop_cycler)["color"]
                if max_epoch == 10:
                    style = "--X"
                elif max_epoch == 20:
                    style = ":X"
                elif max_epoch == 50:
                    style = "-X"
            else:
                color = next(ax._get_lines.prop_cycler)["color"]
                if color == "r":
                    color = next(ax._get_lines.prop_cycler)["color"]
                style = "-o"
            print(color)
            means["scaled_dataset_size"] = 100 * means["dataset_size"]
            means.plot(x="scaled_dataset_size", y=metric, ax=ax, label=label,
                        style=style, linewidth=2, markersize=8, color=color)
            color = ax.lines[-1].get_color()
            
            if lin_probe_col is None and "lin_probe" in metric:
                lin_probe_col = color
            
            # ADD ERRORBAR
            means = means[metric].to_numpy()
            lower, upper = means - std, means + std
            sizes = epoch_df["dataset_size"].unique() * 100
            ax.fill_between(sizes, lower, upper, alpha=0.1, color=color)  
            
            ax.set_xscale("log", base=10)    
            # make a grid appear
            ax.grid(True)
            ax.set_xlabel("Dataset size in %")# (Log Scale)")
            ax.set_ylabel(f"{mode[0].upper() + mode[1:]} ROC AUC")
            
            import matplotlib.ticker as mticker
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        
    return ax
 