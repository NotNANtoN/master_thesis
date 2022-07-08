import os

import pandas as pd
import hydra
import pytorch_lightning


def setup_cfg(cfg):
    cfg = dict(cfg)
        
    # write above statement but all values are set in cfg
    if cfg["model_name"] == "densenet_224":
        cfg["lr"] = 0.001
        cfg["weight_decay"] = 0.0
        # augs
        cfg["rot_aug"] = 45
        cfg["shift_aug"] = 0.15
        cfg["scale_aug"] = 0.1
        # input format
        #cfg["img_value_scale"] = (1024, 1024)
        #cfg["input_res"] = (224, 224)
        # lr scheduling
        #cfg["lr_scale_when_plateau"] = 0.1
        #cfg["lr_patience"] = 5
        
    # overrides and calculated default vals
    if cfg["batch_size"] is None:
        if cfg["model_name"] in ("ViT-L/14", "RN50x64"):
            batch_size = 8
        elif cfg["model_name"] == "ViT-B/16":
            batch_size = 64
        elif cfg["model_name"] == "RN50":
            batch_size = 128
        elif cfg["model_name"] == "RN50x4":
            batch_size = 128
        elif cfg["model_name"] in ("ViT-B/32", "densenet_224", "densenet_256"):
            batch_size = 128
        else:
            batch_size = 32
        cfg["batch_size"] = batch_size
    cfg["lr"] = float(cfg["lr"])

    # set gpu 
    if cfg["num_gpus"] == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    return cfg


@hydra.main(config_path="conf", config_name="supervised")
def main(cfg):    
    # switch back to original wd using hydra
    os.chdir(hydra.utils.get_original_cwd())
    
    cfg = setup_cfg(cfg)
    
    from supervised_utils import create_save_path_sl
    
    val_list = []
    test_list = []
    for seed in range(cfg["seed"], cfg["seed"] + cfg["num_seeds"]):
        cfg["seed"] = seed
        
        save_path = create_save_path_sl(cfg, "results/supervised")
        val_metrics, test_metrics = train_and_eval(cfg, save_path)

        val_list.append(val_metrics)
        test_list.append(test_metrics)
    if cfg["num_seeds"] > 1:
        # save in agg folder
        save_path = create_save_path_sl(cfg, "results/agg_supervised")
        os.makedirs(save_path, exist_ok=True)
        val_df = pd.concat(val_list, axis=0)
        test_df = pd.concat(test_list, axis=0)
        val_df.to_csv(save_path + "/val_metrics.csv")
        test_df.to_csv(save_path + "/test_metrics.csv")
        # average over seeds
        mean_val = val_df.mean()
        mean_val.name = "mean"
        mean_test = test_df.mean()
        mean_test.name = "mean"
        # calc std
        std_val = val_df.std()
        std_val.name = "std"
        std_test = test_df.std()
        std_test.name = "std"
        # merge std and mean in new dataframe with correct column names
        mean_std_df_val = pd.concat([mean_val, std_val], axis=1)
        mean_std_df_test = pd.concat([mean_test, std_test], axis=1)
        
        mean_std_df_val.to_csv(os.path.join(save_path, "mean_std_val_metrics.csv"), index=False)
        mean_std_df_test.to_csv(os.path.join(save_path, "mean_std_test_metrics.csv"), index=False)

    
    
def train_and_eval(cfg, save_path):
    from supervised_utils import train_sl_model
    model, dm = train_sl_model(cfg)
    
    from supervised_eval_utils import eval_supervised_model
    val_metrics, test_metrics = eval_supervised_model(model, dm, save_path)
    
    return val_metrics, test_metrics


if __name__ == "__main__":
    main()
