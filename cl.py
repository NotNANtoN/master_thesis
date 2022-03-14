import os

import hydra

wd = os.getcwd()


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    import os
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
    cfg["val_check_interval"] = int(cfg["val_check_interval"] * (32 / cfg["batch_size"]))
    # switch to old wd
    os.chdir(wd)

    # set gpu 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    # seed everything using pytorch lightning
    import pytorch_lightning
    pytorch_lightning.seed_everything(cfg["seed"])
    
    # to fix issue with "Too many files open"
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # import here because we need to set the gpu before importing torch
    from contrastive_learning_utils import init_dm, init_lit_model, init_clip_model, init_trainer, init_test_dms
    root_folder = "/raid/8wiehe/"
    
    clip_base_model, transform, clip_name = init_clip_model(cfg)
    data_module = init_dm(cfg["dataset_name"], root_folder, clip_base_model, transform, cfg, use_cl=True, use_augs=cfg["use_augs"], batch_size=cfg["batch_size"])
    test_data_modules = init_test_dms(["covidx"], root_folder, transform, cfg)
    lit_model = init_lit_model(clip_base_model, test_data_modules, data_module.steps_per_epoch, data_module.label_names, cfg)
    trainer = init_trainer(root_folder, cfg)
    trainer.fit(lit_model, data_module)


if __name__ == "__main__":
    main()
