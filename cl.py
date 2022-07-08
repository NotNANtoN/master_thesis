import os
import copy

import numpy as np
import hydra
import pandas as pd


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_and_save_zero_shot_results(metric_dict, save_path, name=""):
    zero_shot_auc_keys = [key for key in metric_dict.keys() if "zero_shot_auc" in key]
    zero_shot_aucs = {key: metric_dict[key] for key in zero_shot_auc_keys}
    zero_shot_aucs_df = pd.DataFrame(zero_shot_aucs, index=[0])
    zero_shot_aucs_df.to_csv(os.path.join(save_path, f"zero_shot_aucs_{name}.csv"), index=False)


def ext_test(lit_model, cfg, save_path, root_folder, transform_basic):
    from learning_utils import init_test_dms
    from contrastive_learning_utils import test_performance
    
    test_dataset_names = ["covidx", "rsna", "chexpert"] #["covidx", "mimic-cxr", "rsna", "chexpert"]
    test_data_modules = init_test_dms(test_dataset_names, root_folder, transform_basic, cfg)
    
    for name, test_data_module in zip(test_dataset_names, test_data_modules):
        print("Testing on dataset:", name)
        
        metric_dict = test_performance(lit_model, test_data_module)
        ext_val_aucs, ext_test_aucs = metric_dict["val_aucs"], metric_dict["test_aucs"]
        mean_val_auc = np.mean(list(ext_val_aucs.values()))
        mean_test_auc = np.mean(list(ext_test_aucs.values()))
        # save linear probe results to disk
        lin_probe_df = pd.DataFrame({"mean_val_auc": [mean_val_auc],
                                    "mean_test_auc": [mean_test_auc],})
        lin_probe_df.to_csv(os.path.join(save_path, f"lin_probe_results_{name}.csv"), index=False)
        
        extract_and_save_zero_shot_results(metric_dict, save_path, name=name)
        
        
def train_sl_on_cl_model(mode, weights, orig_cfg, save_path, data_module):
    import torch
    torch.cuda.empty_cache()
    
    print("Evaluating mode:", mode)
    sl_save_path = os.path.join(save_path, "sl_" + mode)
    os.makedirs(sl_save_path, exist_ok=True)
    # setup config for supervised learning
    cfg = copy.deepcopy(orig_cfg)
    cfg["max_epochs"] = orig_cfg["sl_max_epochs"]
    if orig_cfg["sl_dataset_size"] is not None:
        cfg["dataset_size"] = orig_cfg["sl_dataset_size"]
    # make sure no gradients are accumulated
    cfg["grad_acc"] = 1
    cfg["adjust_grad_acc_to"] = None
    cfg["num_gpus"] = 1
    # set lr and batch size for supervised learning depending on model and mode
    if orig_cfg["model_name"] == "ViT-B/32":
        if "adapters" in mode:#orig_cfg["mode"]:
            cfg["lr"] = 1e-4
            cfg["batch_size"] = 392
        elif mode == "full": #orig_cfg["mode"] == "full":
            cfg["lr"] = 1e-5
            cfg["batch_size"] = 256
        else:
            raise NotImplementedError(f"Unkown mode for {orig_cfg['model_name']}: {orig_cfg['mode']}")
    elif orig_cfg["model_name"] == "ViT-L/14":
        if "adapters" in mode:
            cfg["lr"] = 3e-5
            cfg["batch_size"] = 8
        elif mode == "full":
            cfg["lr"] = 3e-6
            cfg["batch_size"] = 8
    # create new base model
    from supervised_utils import init_model
    clip_classifier, transform_basic, transform_aug, name = init_model(orig_cfg)
    clip_base_model = clip_classifier.model.visual
    clip_base_model.name = name

    del clip_classifier

    # set weights to trained weights from CL
    clip_base_model.load_state_dict(copy.deepcopy(weights), strict=False)
    print("Trainable params before setting mode: ", get_num_trainable_params(clip_base_model))
    # set to trainable according to mode - add adapter if mode in cfg is full 
    if cfg["mode"] == "full" and mode == "adapters":
        from adapter_utils import add_adapters_visual
        clip_base_model = add_adapters_visual(clip_base_model, cfg["down_sample_size"],
                                              adapter_flow=cfg["adapter_flow"])
        
    from clip_utils import apply_train_mode
    if mode == "new_adapters":
        apply_train_mode("adapters", clip_base_model)
        # add an extra adapter stack (automatically sets all requires_grad of other adapters to False)
        #clip_base_model.transformer.add_adapter()
        clip_base_model.transformer.add_adapter()    
    else:
        # only use apply_train_mode if we did not add another adapter module
        # -> it is not compatible with multiple adapters (it will activate all adapters instead of last)
        apply_train_mode(mode, clip_base_model)
        
    print("Trainable params after setting mode: ", get_num_trainable_params(clip_base_model))
    # add linear layer at end
    from supervised_models import CLIPClassifier
    sl_model = CLIPClassifier(None, mode, data_module.num_labels, visual=clip_base_model)
    # get dataloader for only imgs and labels
    from clip_utils import FinetuneDataModule
    print("Dataset size: ", cfg["dataset_size"])
    sl_dm = FinetuneDataModule(sl_model, transform_basic, 
                                    dataset_name=cfg["dataset_name"], 
                                    mode=cfg["mode"], 
                                    batch_size=cfg["batch_size"],
                                    augs=transform_aug,
                                    dataset_size=cfg["dataset_size"],
                                    use_cl=False,
                                    seed=cfg["seed"])
    sl_model.label_names = sl_dm.label_names
    cfg["val_check_interval"] = 1.0
    # train!
    from supervised_utils import apply_training
    sl_lit_model = apply_training(sl_model, sl_dm, cfg, verbose=False)
    # evaluate model
    from supervised_eval_utils import eval_supervised_model
    val_metrics, test_metrics = eval_supervised_model(sl_lit_model, sl_dm, sl_save_path)


def eval_cl_model(lit_model, data_module, transform_basic, transform_aug, cfg, save_path, root_folder):    
    # move to cuda 
    lit_model.eval()
    lit_model.cuda()
    
    from contrastive_learning_utils import test_performance
    import numpy as np
    import pandas as pd
    
    # do linear probe on external test set
    if cfg["do_ext"]:
         ext_test(lit_model, cfg, save_path, root_folder, transform_basic)
    
    # do linear probe evaluation on own train set
    metric_dict = test_performance(lit_model, data_module)
    val_aucs, test_aucs = metric_dict["val_aucs"], metric_dict["test_aucs"]
    
    mean_val_auc = np.mean(list(val_aucs.values()))
    mean_test_auc = np.mean(list(test_aucs.values()))
    lit_model = lit_model.cpu()
    # save linear probe results to disk
    lin_probe_df = pd.DataFrame({"mean_val_auc": [mean_val_auc], "mean_test_auc": [mean_test_auc],})
    lin_probe_df.to_csv(os.path.join(save_path, "lin_probe_results.csv"), index=False)
    
    # save zero-shot auc results
    extract_and_save_zero_shot_results(metric_dict, save_path)

    # TODO: save generated images to disk
    
    
    if not cfg["do_sl"]:
        lit_model.cpu()
        return
    
    # get deep copy of weights
    import copy
    lit_model.cpu()
    clip_base_model = lit_model.model.visual
    clip_base_model.name = lit_model.model.name
    weights = copy.deepcopy(clip_base_model.state_dict())    
    # save orig cfg
    orig_cfg = copy.deepcopy(cfg)
    
    
    # delete text transformer
    #del clip_base_model.transformer
    
    if cfg["mode"] == "adapters" and cfg["adapter_flow"] == "easy":
        # if we can easily stack another adapter on top of the existing one then go for it!
        modes = ["full", "adapters", "new_adapters"]
        #modes.append("new_adapters")
    else:
        modes = ["full", "adapters"]
        
    print("Trainable params at start: ", get_num_trainable_params(clip_base_model))
    del clip_base_model
    
    for mode in modes:
        train_sl_on_cl_model(mode, weights, orig_cfg, save_path, data_module)
        
        
def init_cl_model(cfg):
    from clip_utils import load_clip
    clip_base_model, transform, name = load_clip(cfg["model_name"], cfg["mode"], 
                                                 device="cpu", down_sample_size=cfg["down_sample_size"],
                                                 adapter_flow=cfg["adapter_flow"])
    if "ViT" in cfg["model_name"]:
        size = 224
    elif cfg["model_name"] == "RN50x4":
        size = 288
    elif cfg["model_name"] == "RN50" or cfg["model_name"] == "RN101":
        size = 224
    else:
        size = 224
        
    if not cfg["pretrained"]:
        clip_base_model.initialize_parameters()
    from supervised_utils import make_224_transforms
    normalize = transform.transforms[-1]
    transform_basic, transform_aug = make_224_transforms(cfg["rot_aug"], 
                                                        cfg["shift_aug"],
                                                        cfg["scale_aug"],
                                                        normalize,
                                                        size)
    return clip_base_model, transform_basic, transform_aug


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
    
    if cfg["num_gpus"] == 1:
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
    from learning_utils import init_trainer
    root_folder = "/raid/8wiehe/"

    from clip_utils import FinetuneDataModule    
    
    clip_base_model, transform_basic, transform_aug = init_cl_model(cfg)
    
    measure_params = False
    if measure_params:
        def get_num_trainable_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        def get_total_num_params(model):
            return sum(p.numel() for p in model.parameters())
        # print number of parameters of model
        if cfg["model_name"] == "densenet":
            print("Number of parameters: ", get_total_num_params(clip_base_model))
        else:
            print("Number of parameters visual: ", get_total_num_params(clip_base_model.visual))
            print("Number of trainable parameters visual: ", get_num_trainable_params(clip_base_model.visual))
        quit()
    
    data_module = FinetuneDataModule(clip_base_model, transform_basic, 
                                    dataset_name=cfg["dataset_name"], 
                                    mode=cfg["mode"], 
                                    batch_size=cfg["batch_size"],
                                    augs=transform_aug,
                                    use_cl=True,
                                    dataset_size=cfg["dataset_size"],
                                    randomize_order=cfg["randomize_order"],
                                    seed=cfg["seed"],)
    
    lit_model = init_lit_model(clip_base_model, data_module.steps_per_epoch, data_module.label_names, cfg)
    lit_model.data_module = data_module
    cfg["val_check_interval"] = 1.0 # data_module.steps_per_epoch #min(int(cfg["val_check_interval"] * (32 / cfg["batch_size"])), len(data_module.train_dataloader()))
    trainer = init_trainer(root_folder, cfg, "cl_early_tests", num_sanity_val_steps=0)
    # fit CL model
    try:
        trainer.fit(lit_model, data_module)
    finally:
        import wandb
        wandb.finish(quiet=True)
    
    
    from supervised_utils import create_save_path_sl
    save_path = create_save_path_sl(cfg, "results/cl")
    
    # save CL model to disk
    #trainer.save_checkpoint(os.path.join(save_path, "model.pth"))
        
        
    eval_cl_model(lit_model, data_module, transform_basic, transform_aug, 
                  cfg, save_path, root_folder)


if __name__ == "__main__":
    main()
