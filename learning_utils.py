import torch
import pytorch_lightning

from clip_utils import load_clip, FinetuneDataModule


def print_param_count(model):
    count = 0
    total_count = 1
    for n, p in model.named_parameters():
        if "adapter" in n:
            #p.requires_grad = False
        #if p.requires_grad:
            #print(n, p.shape)
            pass

        if p.requires_grad:
            count += p.numel()
            #print(n)
        else:
            total_count += p.numel()    
    print(count, total_count, count / total_count)


def init_clip_model(cfg):
    clip_base_model, transform, clip_name = load_clip(cfg["model_name"], cfg["mode"], device="cpu")
    return clip_base_model, transform, clip_name


def init_dm(dataset_name, root_folder, clip_base_model, transform, cfg, use_cl, use_augs, batch_size,
            num_workers=8, dataset_size=None, pin_memory=False):
    data_module = FinetuneDataModule(clip_base_model, transform, dataset_name=dataset_name, mode=cfg["mode"], 
                                use_cl=use_cl, batch_size=batch_size,
                                root_folder=root_folder, 
                                num_workers=num_workers, dataset_size=dataset_size, pin_memory=pin_memory,
                                sent_frac=cfg["sent_frac"],
                                )
    return data_module


def init_test_dms(names, root_folder, transform, cfg):
    return [init_dm(dataset_name, root_folder, None, transform, cfg, use_cl=False,
                    use_augs=False, batch_size=cfg["batch_size"] * 2,
                    num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
                    dataset_size=cfg["dataset_size"] if dataset_name == "mimic-cxr" else None)
            for dataset_name in names]


def init_trainer(root_folder, cfg, project_name, num_sanity_val_steps=1):
    if cfg["debug"]:
        wandb_logger = None
    else:
        wandb_logger = pytorch_lightning.loggers.WandbLogger(name=None, 
                                                            save_dir=root_folder + "pytorch_lightning/", 
                                                            offline=False, id=None, 
                                            anonymous=None, version=None, project=project_name, 
                                            log_model=False, experiment=None)
        wandb_logger.log_hyperparams(cfg)
                

    if "adjust_grad_acc_to" in cfg and cfg["adjust_grad_acc_to"] is not None:
        old_bs = cfg["adjust_grad_acc_to"]
        assert old_bs > cfg["batch_size"], "Batch size to adjust grad acc to must be greater than currently used batch size"
        grad_acc = old_bs // cfg["batch_size"]
    else:
        grad_acc = cfg["grad_acc"]
        
        
    if cfg["num_gpus"] > 1 and cfg["strategy"] == "ddp":
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = cfg["strategy"]
        
    trainer = pytorch_lightning.Trainer(val_check_interval=cfg["val_check_interval"],
                                        precision=cfg["precision"],
                                        logger=wandb_logger,
                                        max_epochs=cfg["max_epochs"],
                                        max_steps=cfg["max_steps"],
                                        gpus=int(torch.cuda.is_available()) * cfg["num_gpus"],
                                        benchmark=True,
                                        #limit_train_batches=0.2,
                                        num_sanity_val_steps=num_sanity_val_steps,
                                        strategy=cfg["strategy"],
                                        accumulate_grad_batches=grad_acc,
                                        )
    return trainer