import torch
import torch.nn.functional as F
from PIL import Image
import clip
import pytorch_lightning
import numpy as np
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger

from clip_utils import load_clip, FinetuneDataModule, apply_train_mode


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
    clip_base_model, transform, clip_name = load_clip(cfg["model_name"], device="cpu")
    return clip_base_model, transform, clip_name


def init_dm(dataset_name, root_folder, clip_base_model, transform, cfg, use_cl, use_augs, batch_size, num_workers=8):
    data_module = FinetuneDataModule(clip_base_model, transform, dataset_name=dataset_name, mode=cfg["mode"], 
                                 use_augs=use_augs, use_cl=use_cl, sent_frac=cfg["sent_frac"], batch_size=batch_size,
                                root_folder=root_folder, use_ffcv=cfg["use_ffcv"], num_workers=num_workers)
    return data_module

def init_test_dms(names, root_folder, transform, cfg):
    return [init_dm(dataset_name, root_folder, None, transform, cfg, use_cl=False,
                    use_augs=False, batch_size=cfg["batch_size"] // 2, num_workers=2)
            for dataset_name in names]

def init_lit_model(clip_base_model, test_data_modules, steps_per_epoch, label_names, cfg):
    lit_model = LitCLCLIP(clip_base_model, test_data_modules,
                 cfg["mode"], cfg["max_epochs"], cfg["lr"], steps_per_epoch, 
                 weight_decay=cfg["weight_decay"], gen_freq=cfg["gen_freq"], use_ffcv=cfg["use_ffcv"],
                 text2img_num_steps=cfg["text2img_num_steps"],
                 text2img_num_feats=cfg["text2img_num_feats"],
                 train_mode=cfg["mode"],
                 mixup_alpha=cfg["mixup_alpha"],
                 mixup_labels=cfg["mixup_labels"],
                 add_noise_level=cfg["add_noise_level"],
                 mult_noise_level=cfg["mult_noise_level"],)
    lit_model.label_names = label_names

    print_param_count(lit_model)
    return lit_model


def init_trainer(root_folder, cfg):
    if cfg["debug"]:
        wandb_logger = None
    else:
        wandb_logger = pytorch_lightning.loggers.WandbLogger(name=None, 
                                                            save_dir=root_folder + "pytorch_lightning/", 
                                                            offline=False, id=None, 
                                            anonymous=None, version=None, project="cl_early_tests", 
                                            log_model=False, experiment=None, prefix='')
        wandb_logger.log_hyperparams(cfg)

        # log gradients and model topology
        #wandb_logger.watch(lit_model)

    trainer = pytorch_lightning.Trainer(val_check_interval=cfg["val_check_interval"],
                                        precision=cfg["precision"],
                                        logger=wandb_logger,
                                        max_epochs=cfg["max_epochs"],
                                        gpus=int(torch.cuda.is_available()),
                                        benchmark=True,
                                        #limit_train_batches=0.2,
                                        )

    if cfg["use_ffcv"]:
        # for ffcv
        from types import MethodType
        import ffcv_custom_PTL_methods 
        trainer.fit_loop.epoch_loop.on_run_start = MethodType(ffcv_custom_PTL_methods.on_run_start, trainer.fit_loop.epoch_loop)
        trainer.fit_loop.epoch_loop.advance = MethodType(ffcv_custom_PTL_methods.advance, trainer.fit_loop.epoch_loop)

    return trainer



def mixup(a, b, alpha):
    return a * (1 - alpha) + b * alpha


def add_nfm_noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    # noise function for noisy feature mixup
    add_noise = 0.0
    mult_noise = 1.0
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.empty_like(x).normal_()
    if mult_noise_level > 0.0:
        mult_noise = mult_noise_level * np.random.beta(2, 5) * (2 * torch.empty_like(x).uniform_() - 1) + 1 
    return mult_noise * x + add_noise



class LitCLCLIP(pytorch_lightning.LightningModule):
    def __init__(self, model, test_dms, mode, max_epochs, learning_rate, steps_per_epoch, 
                 weight_decay=0.2,
                 gen_freq=5,  # after how many val epochs there should be an image generation
                 use_ffcv=False,
                 text2img_num_steps=200,
                 text2img_num_feats=4,
                 train_mode="adapters",
                 mixup_alpha=0.5,
                 mixup_labels=False,
                 add_noise_level=0.4,
                 mult_noise_level=0.2,
                ):
        super().__init__()
        # args
        self.test_dms = test_dms
        self.mode = mode
        self.save_hyperparameters()
        self.max_epochs = max_epochs
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps_per_epoch = steps_per_epoch
        self.gen_freq = gen_freq
        self.text2img_num_steps = text2img_num_steps
        self.text2img_num_feats = text2img_num_feats
        self.train_mode = train_mode
        self.mixup_alpha = mixup_alpha
        self.mixup_labels = mixup_labels
        self.add_noise_level = add_noise_level
        self.mult_noise_level = mult_noise_level
        self.use_ffcv = use_ffcv
        self.sent_frac = 0.8
        
        self.gen_count = 0
        # loss func
        self.loss_func = torch.nn.CrossEntropyLoss()
        
        if self.train_mode == "adapters":
            from adapter_utils import add_adapters
            self.model = add_adapters(self.model)

        # set some params to non-trainable depending on mode
        apply_train_mode(mode, self.model)

        # for metrics
        self.neutral_prompt = "An X-ray scan of the chest of a person"
                
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        imgs, labels, tokenized_texts = batch
        
        out = self.calc_loss(imgs, tokenized_texts, mode="train")
        loss, img_loss, text_loss, img_features, text_features, logits_per_image, logits_per_text = out

        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        lr = self.scheduler.get_last_lr()[0]
        self.log('lr', lr, on_epoch=False, on_step=True, prog_bar=False, logger=True)
        return loss
    
    def on_validation_start(self, *args, **kwargs):
        super().on_validation_start(*args, **kwargs)
        # calculate label encodings for zero-shot eval
        self.label_feats, self.neutral_feats = self.calculate_label_encodings(self.neutral_prompt, self.label_names)

    @torch.no_grad()
    def calculate_label_encodings(self, neutral_prompt, label_names):
        label_prompts = [neutral_prompt + " with " + label for label in label_names]
        tokenized_neutral = clip.tokenize([neutral_prompt]).to(self.device)
        tokenized_labels = clip.tokenize(label_prompts).to(self.device)
        label_feats = F.normalize(self.model.encode_text(tokenized_labels), dim=-1)
        neutral_feats = F.normalize(self.model.encode_text(tokenized_neutral), dim=-1)
        return label_feats, neutral_feats
            
    
    def validation_step(self, batch, batch_idx):
        imgs, labels, tokenized_texts = batch
        
        out = self.calc_loss(imgs, tokenized_texts, mode="val")
        loss, img_loss, text_loss, img_features, text_features, logits_per_image, logits_per_text = out
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        #self.log('val_img_loss', img_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        #self.log('val_text_loss', text_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        
        # calc zero-shot accuracy
        # first calculate similarities between images and neutral, and images and positive prompts
        # log accuracy first per class then macro-averaged
        zero_shot_class_accuracies = self.calc_zero_shot_acc(self.neutral_feats, self.label_feats, img_features, labels, log=True)
        acc_dict = {"val_acc/" + self.label_names[i]: zero_shot_class_accuracies[i]
                    for i in range(len(zero_shot_class_accuracies))}
        self.log_dict(acc_dict, on_epoch=True)
        self.log('val_zero_shot_acc_macro', zero_shot_class_accuracies.mean(), on_epoch=True, prog_bar=True, logger=True)
                
        # calc retrieval at K
        img_retrieval = (torch.argmax(logits_per_image, dim=1) == torch.arange(labels.shape[0], device=labels.device)).float().mean()
        text_retrieval = (torch.argmax(logits_per_text, dim=1) == torch.arange(labels.shape[0], device=labels.device)).float().mean()
        mean_retrieval = (img_retrieval + text_retrieval) / 2
        #self.log('val_retrieval_img', img_retrieval, on_epoch=True, prog_bar=False, logger=True)
        #self.log('val_retrieval_text', text_retrieval, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_retrieval', mean_retrieval, on_epoch=True, prog_bar=True, logger=True)        
        
        return img_features, labels

    def calc_zero_shot_acc(self, neutral_feats, label_feats, img_feats, labels, log=False):
        neutral_sim = torch.cosine_similarity(neutral_feats, img_feats)
        label_sims = torch.stack([torch.cosine_similarity(label_feat.unsqueeze(0), img_feats) for label_feat in label_feats])
        binary_preds = torch.stack([label_sim > neutral_sim for label_sim in label_sims], dim=1).int()
        zero_shot_class_accuracies = (labels.int() == binary_preds).float().mean(dim=0)
        if log:
            self.log('val_neutral_mean_sim', neutral_sim.mean(), on_epoch=True, prog_bar=False, logger=True)
            self.log('val_label_mean_sim', label_sims.mean(), on_epoch=True, prog_bar=False, logger=True)
            self.log('val_label_max_sim', label_sims.max(), on_epoch=True, prog_bar=False, logger=True)
        return zero_shot_class_accuracies
    
    def validation_epoch_end(self, val_step_output_list):
        if self.trainer.sanity_checking:
            return
        # calculate few-shot linear probe
        #print(len(val_step_output_list))  # 2 - num batches (2 for sanity check)
        #print(len(val_step_output_list[0]))  # 2 - num outputs
        #print(len(val_step_output_list[0][0]))  # 32 - batch size
        img_features = F.normalize(torch.cat([step_out[0] for step_out in val_step_output_list]), dim=-1).cpu().float().numpy()
        labels = torch.cat([step_out[1] for step_out in val_step_output_list]).cpu().numpy()
        
        # calculate retrieval at K
        for k in [1, 5, 10, 20]:
            self.k_shot_performance(img_features, labels, k=k)
            
    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
    

    def on_train_epoch_start(self) -> None:
        # calculate performance on external test sets!
        if self.trainer.sanity_checking:
            return
        for test_dm in self.test_dms:
            self.test_performance(test_dm)
        self.model.train()
        
        # generate image from text here for some labels - need to enable gradient for it
        #if self.gen_count % self.gen_freq == 0 and self.text2img_num_feats > 0:
            
        with torch.set_grad_enabled(True):
                self.gen_img_from_label()
        #self.gen_count += 1
        
        return super().on_epoch_start()

    @torch.no_grad()
    def get_feats(self, dl):
        # get image features and labels
        img_features = []
        labels = []
        for batch in dl:
            imgs, batch_labels = batch
            feats = F.normalize(self.model.encode_image(imgs.cuda()), dim=-1)
            img_features.append(feats.cpu())
            labels.append(batch_labels.cpu())
        img_features = torch.cat(img_features)
        labels = torch.cat(labels)
        return img_features, labels

    def test_performance(self, test_dm: FinetuneDataModule):
        name = test_dm.dataset_name 

        # calculate performance on external test dataset
        self.model.eval()

        # create dataloaders
        #test_dl = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        train_dl, test_dl = test_dm.train_dataloader(), test_dm.test_dataloader()

        # get image features and labels
        train_img_features, train_labels = self.get_feats(train_dl)
        test_img_features, test_labels = self.get_feats(test_dl)

        # calculate and log zero-shot accuracy
        label_names = test_dm.label_names
        label_feats, neutral_feats = self.calculate_label_encodings(self.neutral_prompt, label_names)
        label_feats, neutral_feats = label_feats.cpu(), neutral_feats.cpu()
        zero_shot_class_accuracies = self.calc_zero_shot_acc(neutral_feats, label_feats, test_img_features, test_labels, log=False)
        self.log(f"{name}/zero_shot_acc_macro", zero_shot_class_accuracies.mean(), on_epoch=True, prog_bar=True, logger=True)

        # calculate and log linear probe auc using logistic regression
        aucs = []
        for i in range(train_labels.shape[-1]):
            train_feat_labels = train_labels[:, i]
            test_feat_labels = test_labels[:, i]

            # import and fit logistic regression
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=500)
            lr.fit(train_img_features, train_feat_labels)
            test_preds = lr.predict_proba(test_img_features)
            test_preds = test_preds[:, 1]

            # calculate auc
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(test_feat_labels, test_preds)
            aucs.append(auc)
            label_name = label_names[i]
            self.log(f"{name}/{label_name}_auc", auc, on_epoch=True, prog_bar=True, logger=True)
            
        self.log(f"{name}/auc_macro", np.mean(aucs), on_epoch=True, prog_bar=True, logger=True)
    

    def k_shot_performance(self, img_features, labels, k=5):
        # create few-shot datasets for each label
        accs = []
        aucs = []
        #print("K:", k)
        for i in range(labels.shape[-1]):
            feat_labels = labels[:, i]
            # get few-shot idcs
            # get masks of pos/neg labels
            true_mask = feat_labels == 1
            false_mask = feat_labels == 0
            
            #print(i, true_mask.sum())
            if true_mask.sum() == 0:
                print("No one in val set: ", self.label_names[i])
                continue
            
            # do not re-use indices we added before (only needed if we predict multiple labels at once)
            #true_mask[support_idcs] = 0
            #false_mask[support_idcs] = 0
            
            # get idcs
            pos_idcs = np.where(true_mask)[0][:k]            
            neg_idcs = np.where(false_mask)[0][:k]
            # get support/train and test idcs
            support_idcs = np.concatenate([pos_idcs, neg_idcs]).reshape(-1)
            test_idcs = [i for i in range(feat_labels.shape[0]) if i not in set(support_idcs)]
        
            # get data splits
            train_data = img_features[support_idcs]
            train_labels = feat_labels[support_idcs]
            test_data = img_features[test_idcs]
            test_labels = feat_labels[test_idcs]
            # train linear probe
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
            model.fit(train_data, train_labels)
            
            # make pred
            pred_labels = model.predict(test_data)
            # calc metrics
            from sklearn.metrics import accuracy_score, roc_auc_score
            acc = accuracy_score(test_labels, pred_labels)
            auc = roc_auc_score(test_labels, pred_labels)
            # save and log metrics
            accs.append(acc)
            aucs.append(auc)
            #self.log(f"val_{k}_shot_acc/{self.label_names[i]}", acc)
            #self.log(f"val_{k}_shot_auc/{self.label_names[i]}", auc)

        # log mean acc and auc
        macro_acc = np.mean(accs)
        macro_auc = np.mean(aucs)
        self.log(f"val_{k}_shot_macro_acc", macro_acc)
        self.log(f"val_{k}_shot_macro_auc", macro_auc)

    
    def calc_loss(self, imgs, tokenized_texts, mode='train'):
        bs = imgs.shape[0]
        image_features = F.normalize(self.model.encode_image(imgs), dim=-1)
        text_features = F.normalize(self.model.encode_text(tokenized_texts), dim=-1)
        logit_scale = self.model.logit_scale.exp()

        # do mixup and add noise if desired in train mode
        if mode == 'train':
            if self.mixup_alpha > 0:
                # sample mixup strength uniformly
                mixup_strength = np.random.uniform(0.0, self.mixup_alpha)
                # get mixup idcs
                mixup_indices = torch.randperm(bs, device=imgs.device, dtype=torch.long)
                # mix features
                image_features = mixup(image_features, image_features[mixup_indices], mixup_strength)
                text_features = mixup(text_features, text_features[mixup_indices], mixup_strength)

            # add noise to features
            if self.add_noise_level > 0 or self.mult_noise_level > 0:
                image_features = add_nfm_noise(image_features, self.add_noise_level, self.mult_noise_level)
                text_features = add_nfm_noise(text_features, self.add_noise_level, self.mult_noise_level)
        # mix labels if desired when using mixup in train mode
        if mode == "train" and self.mixup_alpha > 0 and self.mixup_labels:
            # generate labels
            ground_truth = torch.zeros(bs, bs, dtype=torch.float, device=imgs.device)
            ground_truth[torch.arange(bs), mixup_indices] = mixup_strength
            ground_truth[torch.arange(bs), torch.arange(bs)] = 1 - mixup_strength
        else:
            # define labels
            ground_truth = torch.arange(bs, dtype=torch.long, device=imgs.device)
        # cosine similarity as logits
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        # shape = [global_batch_size, global_batch_size]
        
        img_loss = self.loss_func(logits_per_image, ground_truth)
        text_loss = self.loss_func(logits_per_text, ground_truth)
        loss = (img_loss + text_loss) / 2
        return loss, img_loss, text_loss, image_features, text_features, logits_per_image, logits_per_text
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                              lr=self.learning_rate,#5e-5,
                              betas=(0.9, 0.98),
                              eps=1e-6,
                              weight_decay=self.weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.learning_rate, steps_per_epoch=self.steps_per_epoch, epochs=self.max_epochs, pct_start=0.05)
        scheduler = {"scheduler": self.scheduler, 
                     "interval": "step" }  # necessary to make PL call the scheduler.step after every batch
        return [optimizer], [scheduler]
    
    def gen_img_from_label(self):
        if self.logger is None:
            return
        
        # init text2img
        import sys
        sys.path.append("../CLIPGuidance/")
        from style_clip import Imagine
        
        net = "image"
        size = 256
        if net == "dip":
            kwargs = {"model_type": "dip",
                      "lr": 0.0003,
                      "dip_num_scales": 5,
                      "stack_size": 3,
                     }
        elif net == "image":
            kwargs = {"model_type": "image",
                      "lr": 0.03,
                      "stack_size": 4,
                     }
        elif net == "vqgan":
            kwargs = {"model_type": "vqgan",
                      "lr": 0.1,
                      "stack_size": 1,
                     }
        for p in self.model.parameters():
            p.requires_grad_(False)
        imagine = Imagine(save_gif=False, save_video=False, save_progress=False, verbose=False,
                                  sideX = size, 
                                  sideY = size,
                                  batch_size=8,
                                  clip_models=[self.model],
                                  #clip_names=('ViT-B/16',),
                                  grayscale=True,
                                  tv_loss_scale=0.0,
                                  seed=42,
                                  use_russell_transform=True,
                                  use_mixed_precision=True,
                                  optimizer="AdamW", #"MADGRAD",
                                  **kwargs
                                 )
        
        # create imgs
        used_names = self.label_names[:self.text2img_num_feats]
        imgs = []
        for label in used_names:
            imagine.reset()
            label_prompt = self.neutral_prompt + " with " + label
            imagine.set_clip_encoding(text=label_prompt) #"FINDING: The patient has a " + label)
            for i in range(self.text2img_num_steps):
                img, _ = imagine.train_step(0, 0)
            imgs.append(img.squeeze().detach().cpu().float())
        
        easy_log = True
        
        if easy_log:
            self.logger.log_image(key="labels2img", images=imgs, caption=used_names)
        else:
            imgs = torch.stack(imgs, dim=0).cpu()
            # log to wandb
            import torchvision
            image_array = torchvision.utils.make_grid(imgs, nrow=5, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0)
            import wandb
            images = wandb.Image(image_array, caption=", ".join(used_names))
            self.logger.log({"labels2img": images})

        # re-enable grads for CLIP model (were disabled earlier in text2img because we aimed at optimizing the input image)
        apply_train_mode(self.mode, self.model)
