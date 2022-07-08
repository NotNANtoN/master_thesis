import os

import torch
import torchvision
import torchvision.transforms as TF

from supervised_models import CLIPClassifier


def create_save_path_sl(cfg, root_folder):
    # return path to folder where metrics will be saved that includes current datetime
    # get current datetime as string
    import datetime
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    # get scientific notation of lr
    lr_str = "{:.0e}".format(cfg["lr"])
    path = f"{root_folder}/{now_str}_{cfg['model_name'].replace('/','_')}_{lr_str}_{str(cfg['dataset_size']).replace('.', '_')}_{cfg['seed']}/"
    os.makedirs(path, exist_ok=True)
    # save cfg there as json
    import json
    with open(path + "cfg.json", "w") as f:
        json.dump(cfg, f)
    return path


def apply_training(base_model, data_module, cfg, verbose=True):
    base_model.label_names = data_module.label_names
    
    # import here because we need to set the gpu before importing torch
    root_folder = "/raid/8wiehe/"
    from learning_utils import init_trainer#, init_test_dms
    
    from supervised_models import LitCLIP
    lit_model = LitCLIP(base_model, data_module.label_names,
                        cfg["max_epochs"], cfg["lr"], data_module.steps_per_epoch, 
                        weight_decay=cfg["weight_decay"], use_pos_weight=cfg["use_pos_weight"],
                        pos_fraction=data_module.pos_fraction, eps=cfg["eps"], beta1=cfg["beta1"], beta2=cfg["beta2"])
    
    #cfg["val_check_interval"] = min(int(cfg["val_check_interval"] * (32 / cfg["batch_size"])), len(data_module.train_dataloader()))
    if verbose:
        cfg["val_check_interval"] = 0.1 if cfg["dataset_size"] > 0.1 else 0.5
    else:
        cfg["val_check_interval"] = data_module.steps_per_epoch
    trainer = init_trainer(root_folder, cfg, "early_tests", num_sanity_val_steps=0)
    trainer.fit(lit_model, data_module)
    
    import wandb
    wandb.finish(quiet=not verbose)
    
    return lit_model

def train_sl_model(cfg):
    # to fix pytorch issue with "Too many files open"
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    # seed everything using pytorch lightning
    import pytorch_lightning
    pytorch_lightning.seed_everything(cfg["seed"])
    
    # init model
    #from supervised_utils import init_model
    base_model, transform_basic, transform_aug, name = init_model(cfg)

    from clip_utils import FinetuneDataModule
    data_module = FinetuneDataModule(base_model, transform_basic, 
                                dataset_name=cfg["dataset_name"], 
                                mode=cfg["mode"], 
                                batch_size=cfg["batch_size"],
                                augs=transform_aug,
                                use_cl=False,
                                dataset_size=cfg["dataset_size"],)
    
    lit_model = apply_training(base_model, data_module, cfg)
    
    return lit_model, data_module
    

def make_224_transforms(rot_aug, shift_aug, scale_aug, normalize, size):
    transform_basic = TF.Compose([TF.Resize(size=size, 
                                    interpolation=TF.InterpolationMode.BILINEAR),
                                TF.CenterCrop(size=(size, size)),
                                TF.ToTensor(),
                                normalize])
    transform_aug = TF.Compose([TF.Resize(size=size,
                                        interpolation=TF.InterpolationMode.BILINEAR),
                                TF.CenterCrop(size=(size, size)),
                                TF.RandomAffine(rot_aug,
                                                translate=(shift_aug, shift_aug),
                                                scale=(1.0 - scale_aug, 1.0 + scale_aug)),
                                TF.ToTensor(),
                                normalize])
    return transform_basic, transform_aug


def init_model(cfg):
    pretrained = cfg["pretrained"]
    #convert_models_to_fp32(model)
    num_labels = 14 #data_module.num_labels

    if cfg["model_name"] == "densenet_224_cxr":
        name = "densenet_224"
        densenet_size = 224
        import torchxrayvision as xrv
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        
        
        labels_to_remove = ["No finding", "Support devices", "Pleural other"]

        # Use XRV transforms to crop and resize the images
        transform_basic = TF.Compose([xrv.datasets.ToPILImage(),
                                    xrv.datasets.XRayCenterCrop(),
                                    xrv.datasets.XRayResizer(densenet_size)])

        transform_aug = TF.Compose([
            xrv.datasets.ToPILImage(),
            TF.RandomAffine(cfg["rot_aug"], 
                                                translate=(cfg["shift_aug"], cfg["shift_aug"]), 
                                                scale=(1.0 - cfg["scale_aug"], 1.0 + cfg["scale_aug"])),
            TF.ToTensor()
        ])
    elif "densenet" in cfg["model_name"]:
        if cfg["model_name"] == "densenet_256":
            name = "densenet_256"
            densenet_size = 256
            model = torchvision.models.densenet121(pretrained=pretrained) 

        elif cfg["model_name"] == "densenet_224":
            name = "densenet_224"
            densenet_size = 224
            import torchxrayvision as xrv
            model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
            
        # get model and re-init out layer
        model.classifier = torch.nn.Linear(1024, num_labels)
        model.num_labels = num_labels
        # create transform
        normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        
        transform_basic, transform_aug = make_224_transforms(cfg["rot_aug"], 
                                                            cfg["shift_aug"],
                                                            cfg["scale_aug"],
                                                            normalize,
                                                            224)
    else:
        from clip_utils import load_clip
        clip_base_model, transform, name = load_clip(cfg["model_name"], cfg["mode"], device="cpu", 
                                                     down_sample_size=cfg["down_sample_size"], 
                                                     adapter_flow=cfg["adapter_flow"])

        if "ViT" in cfg["model_name"]:
            size = 224
        elif cfg["model_name"] == "RN50x4":
            size = 288
        elif cfg["model_name"] == "RN50" or cfg["model_name"] == "RN101":
            size = 224
        else:
            size = 224
            
        if not pretrained:
            clip_base_model.initialize_parameters()
            
        
        model = CLIPClassifier(clip_base_model, cfg["mode"], num_labels)
        normalize = transform.transforms[-1]
        transform_basic, transform_aug = make_224_transforms(cfg["rot_aug"], 
                                                            cfg["shift_aug"],
                                                            cfg["scale_aug"],
                                                            normalize,
                                                            size)
    model.name = name
    
        
    return model, transform_basic, transform_aug, name