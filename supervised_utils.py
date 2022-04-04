import torch
import torchvision

from supervised_models import CLIPClassifier

def init_model(cfg):
    #convert_models_to_fp32(model)
    num_labels = 14 #data_module.num_labels

    if cfg["model_name"] == "densenet_224":
        densenet_size = 224
        import torchxrayvision as xrv
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        labels_to_remove = ["No finding", "Support devices", "Pleural other"]
        

        # Use XRV transforms to crop and resize the images
        transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                    xrv.datasets.XRayResizer(densenet_size)])
        
        data_aug = torchvision.transforms.Compose([
            xrv.datasets.ToPILImage(),
            torchvision.transforms.RandomAffine(cfg.data_aug_rot, 
                                                translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                                scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
            torchvision.transforms.ToTensor()
        ])
        
    elif cfg["model_name"] == "densenet_256":
        densenet_size = 256
        import torchvision
        import torchvision.transforms as TF
        # get model and re-init out layer
        model = torchvision.models.densenet121(pretrained=True) 
        model.classifier = torch.nn.Linear(1024, num_labels)
        model.num_labels = num_labels
        # create transform
        normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform = TF.Compose([TF.Resize(size=densenet_size, 
                                          interpolation=TF.InterpolationMode.BILINEAR),
                                        TF.CenterCrop(size=(densenet_size, densenet_size)),
                                        TF.ToTensor(),
                                        normalize])
    else:
        from clip_utils import load_clip
        clip_base_model, transform, clip_name = load_clip(model_name, device="cpu")
        model = CLIPClassifier(clip_base_model, mode, num_labels)
        
    return model, transform, transform_aug, name