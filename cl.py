import os

import hydra


@hydra.main(config_path="conf", config_name="config")
def main(cfg):    
    # switch to original wd using hydra
    os.chdir(hydra.utils.get_original_cwd())

    cfg = dict(cfg)
    # overrides and calculated default vals
    if cfg["batch_size"] is None:
        if cfg["model_name"] == "ViT-L/14":
            batch_size = 4 # max 32 for single GPU CL on VitB16, 4 for ViT-L/14 (9.2GB)
        elif cfg["model_name"] == "ViT-B/16":
            batch_size = 32
        elif cfg["model_name"] == "ViT-B/32":
            batch_size = 64
        else:
            batch_size = 32
        cfg["batch_size"] = batch_size
    cfg["lr"] = float(cfg["lr"])
    # set gpu 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    # seed everything using pytorch lightning
    import pytorch_lightning
    pytorch_lightning.seed_everything(cfg["seed"])
    
    # to fix pytorch issue with "Too many files open"
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # import here because we need to set the gpu before importing torch
    from contrastive_learning_utils import init_lit_model
    from learning_utils import init_dm, init_test_dms, init_clip_model, init_trainer
    root_folder = "/raid/8wiehe/"
    
    clip_base_model, transform, clip_name = init_clip_model(cfg)
    data_module = init_dm(cfg["dataset_name"], root_folder, clip_base_model, transform, cfg, use_cl=True,
                          use_augs=cfg["use_augs"], batch_size=cfg["batch_size"], dataset_size=cfg["dataset_size"],
                          num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])



    test_data_modules = init_test_dms(["covidx", "mimic-cxr", "rsna", "chexpert"], root_folder, transform, cfg)
    lit_model = init_lit_model(clip_base_model, test_data_modules, data_module.steps_per_epoch, data_module.label_names, cfg)
    cfg["val_check_interval"] = min(int(cfg["val_check_interval"] * (32 / cfg["batch_size"])), len(data_module.train_dataloader()))
    trainer = init_trainer(root_folder, cfg)
    trainer.fit(lit_model, data_module)
    
    import wandb
    wandb.finish()


if __name__ == "__main__":
    main()
