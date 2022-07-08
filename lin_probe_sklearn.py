import os

import hydra

from linear_probe_utils import binary_relevance_linear_probe


@hydra.main(config_path="conf", config_name="supervised")
def main(cfg):    
    # switch back to original wd using hydra
    os.chdir(hydra.utils.get_original_cwd())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    from linear_probe_utils import setup_linear_probe
    out = setup_linear_probe(cfg)
    cfg, feats, label_names, steps_per_epoch, pos_fraction, folder_name = out
    [train_img_features, train_labels], [val_img_features, val_labels], [test_img_features, test_labels] = feats

    
    # calculate and log linear probe auc using logistic regression
    
    test_aucs, test_aps, val_aucs, val_aps = binary_relevance_linear_probe(label_names, 
                                                                            train_img_features, train_labels ,
                                                                            val_img_features, val_labels,
                                                                            test_img_features, test_labels)
    
    
    save_folder = "results/binary_relevance"
    os.makedirs(os.path.join(save_folder, folder_name), exist_ok=True)
    
    # save results with json
    import json
    with open(os.path.join(save_folder, folder_name, "val_auc.json"), "w") as f:
        json.dump(val_aucs, f)
    with open(os.path.join(save_folder, folder_name, "val_ap.json"), "w") as f:
        json.dump(val_aps, f)
    with open(os.path.join(save_folder, folder_name, "test_auc.json"), "w") as f:
        json.dump(test_aucs, f)
    with open(os.path.join(save_folder, folder_name, "test_ap.json"), "w") as f:
        json.dump(test_aps, f)
        
            
    
    
if __name__ == "__main__":
    main()