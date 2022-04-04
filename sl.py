import os

import hydra


@hydra.main(config_path="conf", config_name="supervised")
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
    
    
        
    # check code of torchxrayvisoin for default learning rate and for num epochs until decay
    
    # write above statement but all values are set in cfg
    if cfg["model_name"] == "densenet_xrv":
        cfg["model_name"] = "densenet_224"
        cfg["lr"] = 0.001
        cfg["weight_decay"] = 0.0
        cfg["batch_size"] = 64
        # augs
        cfg["rot_aug"] = 45
        cfg["shift_aug"] = 0.15
        cfg["scale_aug"] = 0.1
        # input format
        cfg["img_value_scale"] = (1024, 1024)
        cfg["input_res"] = (224, 224)
        # lr scheduling
        cfg["lr_scale_when_plateau"] = 0.1
        cfg["lr_patience"] = 5
    elif cfg["model_name"] == "ViT-L/14":
        cfg["batch_size"] = 8 # max 32 for single GPU CL on VitB16, 16 for ViT-L/14
    elif cfg["model_name"] == "ViT-B/16":
        cfg["batch_size"] = 32
    else:
        batch_size = 32
    
    # set gpu 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    # seed everything using pytorch lightning
    import pytorch_lightning
    pytorch_lightning.seed_everything(cfg["seed"])
    
    # to fix pytorch issue with "Too many files open"
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # init model
    from supervised_utils import init_model
    from clip_utils import FinetuneDataModule
    base_model, transform, name = init_model(cfg["model_name"])

    data_module = FinetuneDataModule(base_model, transform, 
                                    dataset_name=cfg["dataset_name"], 
                                    mode=cfg["mode"], 
                                    use_augs=cfg["use_augs"],
                                    batch_size=cfg["batch_size"])
    base_model.label_names = data_module.label_names
    
    
    # import here because we need to set the gpu before importing torch
    root_folder = "/raid/8wiehe/"
    from learning_utils import init_dm, init_trainer, init_test_dms, init_lit_sl_model
    
    data_module = init_dm(cfg["dataset_name"], root_folder, base_model, transform, cfg, 
                          use_cl=False,
                          use_augs=cfg["use_augs"], batch_size=cfg["batch_size"], dataset_size=cfg["dataset_size"],
                          num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])



    test_data_modules = init_test_dms(["covidx", "mimic-cxr", "rsna", "chexpert"], root_folder, transform, cfg)
    lit_model = init_lit_sl_model(clip_base_model, test_data_modules, data_module.steps_per_epoch, data_module.label_names, cfg)
    cfg["val_check_interval"] = min(int(cfg["val_check_interval"] * (32 / cfg["batch_size"])), len(data_module.train_dataloader()))
    trainer = init_trainer(root_folder, cfg)
    trainer.fit(lit_model, data_module)
    
    import wandb
    wandb.finish()


if __name__ == "__main__":
    main()
