import torch
import torch.nn.functional as F
import clip
import pytorch_lightning
import numpy as np
import pytorch_lightning
from tqdm.auto import tqdm

from clip_utils import FinetuneDataModule, apply_train_mode
from learning_utils import print_param_count
from linear_probe_utils import binary_relevance_linear_probe
   
   
class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


def init_lit_model(clip_base_model, steps_per_epoch, label_names, cfg):
    lit_model = LitCLCLIP(clip_base_model,
                 cfg["mode"], cfg["max_epochs"], cfg["lr"], steps_per_epoch, 
                 weight_decay=cfg["weight_decay"], gen_freq=cfg["gen_freq"],
                 text2img_num_steps=cfg["text2img_num_steps"],
                 text2img_num_feats=cfg["text2img_num_feats"],
                 text2img_batch_size=cfg["text2img_batch_size"],
                 mixup_alpha=cfg["mixup_alpha"],
                 mixup_labels=cfg["mixup_labels"],
                 add_noise_level=cfg["add_noise_level"],
                 mult_noise_level=cfg["mult_noise_level"],
                 cyclic_lambda=cfg["cyclic_lambda"],)
    lit_model.label_names = label_names

    print_param_count(lit_model)
    return lit_model


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


def encode_raw_text(model, device, text):
    tokenized = clip.tokenize(text).to(device)
    feats =  F.normalize(model.encode_text(tokenized), dim=-1)
    return feats


@torch.no_grad()
def calculate_label_encodings(model, device, neutral_prompt, label_names):
    label_prompts = [neutral_prompt + " with " + label for label in label_names]
    label_feats = encode_raw_text(model, device, label_prompts)
    neutral_feats = encode_raw_text(model, device, [neutral_prompt])
    return label_feats, neutral_feats


def calc_zero_shot_acc(neutral_feats, label_feats, img_feats, labels):
    if not torch.is_tensor(img_feats):
        img_feats = torch.tensor(img_feats)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    neutral_sim = torch.cosine_similarity(neutral_feats, img_feats)
    label_sims = torch.stack([torch.cosine_similarity(label_feat.unsqueeze(0), img_feats) for label_feat in label_feats])
    binary_preds = torch.stack([label_sim > neutral_sim for label_sim in label_sims], dim=1).int()
    
    accs = []
    for i in range(labels.shape[1]):
        # select only the i-th column of the binary_preds
        preds = binary_preds[:, i]
        # select only the i-th column of the labels
        labels_i = labels[:, i]
        # filter preds and labels where labels are nan
        preds = preds[~torch.isnan(labels_i)]
        labels_i = labels_i[~torch.isnan(labels_i)]
        # if there are no labels or only one type of label, skip
        if len(labels_i) == 0 or len(torch.unique(labels_i)) == 1:
            continue
        # calculate the accuracy of the i-th column
        acc = (preds == labels_i).float().mean()
        accs.append(acc)
    return torch.stack(accs).cpu()


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def calc_zero_shot_auc(neutral_feats, label_feats, img_feats, labels):
    if not torch.is_tensor(img_feats):
        img_feats = torch.tensor(img_feats)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    # calc cosine similarities to neutral prompt and to labels
    neutral_sim = torch.cosine_similarity(neutral_feats, img_feats)
    label_sims = torch.stack([torch.cosine_similarity(label_feat.unsqueeze(0), img_feats) for label_feat in label_feats])
    # calc difference between cosine similarities to neutral prompt and to labels
    sim_diff = torch.stack([label_sim - neutral_sim for label_sim in label_sims], dim=1).float()
    sim_diff = minmax(sim_diff)
    # calc auc
    from sklearn.metrics import roc_auc_score
    
    aucs = []
    for i in range(labels.shape[1]):
        # select only the i-th column of the binary_preds
        preds = sim_diff[:, i]
        # select only the i-th column of the labels
        labels_i = labels[:, i]
        # filter preds and labels where labels are nan
        preds = preds[~torch.isnan(labels_i)]
        labels_i = labels_i[~torch.isnan(labels_i)]
        # if there are no labels or only one type of label, skip
        if len(labels_i) == 0 or len(torch.unique(labels_i)) == 1:
            continue
        # calculate the auc of the i-th column
        auc = roc_auc_score(labels_i.cpu().numpy(), preds.cpu().numpy())
        aucs.append(auc)
    return np.stack(aucs)
    
    # filter preds and labels where labels are nan
    #sim_diff = sim_diff[torch.isnan(labels) == 0]
    #labels = labels[torch.isnan(labels) == 0]
    #aucs = roc_auc_score(labels.cpu().numpy(), sim_diff.cpu().numpy(), average="macro")   
    #return aucs


def test_performance(model, test_dm: FinetuneDataModule):
    name = test_dm.dataset_name 
    print("Testing dataset:", name)

    # create dataloaders
    #test_dl = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    train_dl, val_dl, test_dl = test_dm.train_dataloader(), test_dm.val_dataloader(), test_dm.test_dataloader()

    # get image features and labels
    from supervised_eval_utils import get_preds
    train_img_features, train_labels = get_preds(model.encode_image, train_dl)# get_feats(self.model, train_dl, use_tqdm=True)
    test_img_features, test_labels = get_preds(model.encode_image, test_dl) #get_feats(self.model, test_dl)
    val_img_features, val_labels = get_preds(model.encode_image, val_dl) #get_feats(self.model, val_dl)
    
    
    # calculate and log linear probe auc using logistic regression
    test_aucs, test_aps, val_aucs, val_aps = binary_relevance_linear_probe(test_dm.label_names, 
                                                train_img_features, train_labels,
                                                val_img_features, val_labels,
                                                test_img_features, test_labels)
            
    metric_dict = {
        "test_aucs": test_aucs,
        "test_aps": test_aps,
        "val_aucs": val_aucs,
        "val_aps": val_aps,
    }    
    
    # calculate and log zero-shot accuracy
    neutral_prompts = ["Impression: ", "FINDINGS:"]
    neutral_feats, all_label_feats = calc_zero_shot_embeddings(model.model, "cuda", neutral_prompts, test_dm.label_names)
    
    neutral_feats = neutral_feats.to("cpu") 
    all_label_feats = [f.to("cpu") for f in all_label_feats]
    # calc zero-shot accuracy
    # first calculate similarities between images and neutral, and images and positive prompts
    for i, prompt in enumerate(neutral_prompts):
        neutral_feats_label = neutral_feats[i]
        label_feats = all_label_feats[i]
        cleaned_prompt = prompt.replace(" ", "_")
                    
        val_zero_shot_macro_acc = calc_zero_shot_acc(neutral_feats_label, label_feats, val_img_features, val_labels).mean()
        test_zero_shot_macro_acc = calc_zero_shot_acc(neutral_feats_label, label_feats, test_img_features, test_labels).mean()
        
        val_zero_shot_macro_auc = calc_zero_shot_auc(neutral_feats_label, label_feats, val_img_features, val_labels).mean()
        test_zero_shot_macro_auc = calc_zero_shot_auc(neutral_feats_label, label_feats, test_img_features, test_labels).mean()
        
        metric_dict[f"val_zero_shot_acc_{cleaned_prompt}"] = val_zero_shot_macro_acc
        metric_dict[f"test_zero_shot_acc_{cleaned_prompt}"] = test_zero_shot_macro_acc
        metric_dict[f"val_zero_shot_auc_{cleaned_prompt}"] = val_zero_shot_macro_auc
        metric_dict[f"test_zero_shot_auc_{cleaned_prompt}"] = test_zero_shot_macro_auc
        
    # calculate zero-shot accuracy for averaged prompts
    averaged_netral_feats = F.normalize(torch.mean(neutral_feats, dim=0), dim=-1)
    averaged_label_feats = F.normalize(torch.mean(torch.stack(all_label_feats, dim=0), dim=0), dim=-1)
    val_averaged_zero_shot_macro_acc = calc_zero_shot_acc(averaged_netral_feats, averaged_label_feats, val_img_features, val_labels).mean()
    test_averaged_zero_shot_macro_acc = calc_zero_shot_acc(averaged_netral_feats, averaged_label_feats, test_img_features, test_labels).mean()
    val_averaged_zero_shot_macro_auc = calc_zero_shot_auc(averaged_netral_feats, averaged_label_feats, val_img_features, val_labels).mean()
    test_averaged_zero_shot_macro_auc = calc_zero_shot_auc(averaged_netral_feats, averaged_label_feats, test_img_features, test_labels).mean()

    metric_dict["val_zero_shot_acc_averaged"] = val_averaged_zero_shot_macro_acc
    metric_dict["test_zero_shot_acc_averaged"] = test_averaged_zero_shot_macro_acc
    metric_dict["val_zero_shot_auc_averaged"] = val_averaged_zero_shot_macro_auc
    metric_dict["test_zero_shot_auc_averaged"] = test_averaged_zero_shot_macro_auc    
    
    return metric_dict


@torch.no_grad()
def calc_zero_shot_embeddings(model, device, neutral_prompts, label_names):
    neutral_feats = encode_raw_text(model, device, neutral_prompts)
    all_label_feats = []
    for neutral_prompt in neutral_prompts:
        label_prompts = [neutral_prompt + " with " + label for label in label_names]
        label_feats = encode_raw_text(model, device, label_prompts)
        all_label_feats.append(label_feats)
    return neutral_feats, all_label_feats
    

class LitCLCLIP(pytorch_lightning.LightningModule):
    def __init__(self, model, mode, max_epochs, learning_rate, steps_per_epoch, 
                 weight_decay=0.2,
                 gen_freq=5,  # after how many val epochs there should be an image generation
                 text2img_num_steps=200,
                 text2img_num_feats=4,
                 text2img_batch_size=16,
                 mixup_alpha=0.5,
                 mixup_labels=False,
                 add_noise_level=0.4,
                 mult_noise_level=0.2,
                 cyclic_lambda=None,
                ):
        super().__init__()
        # args
        self.mode = mode
        self.save_hyperparameters(ignore=['model'])
        self.max_epochs = max_epochs
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps_per_epoch = steps_per_epoch
        self.gen_freq = gen_freq
        self.text2img_num_steps = text2img_num_steps
        self.text2img_num_feats = text2img_num_feats
        self.text2img_batch_size = text2img_batch_size
        self.mixup_alpha = mixup_alpha
        self.mixup_labels = mixup_labels
        self.add_noise_level = add_noise_level
        self.mult_noise_level = mult_noise_level
        self.sent_frac = 0.8
        self.cyclic_lambda = cyclic_lambda
        
        
        self.gen_count = 0
        # loss func
        self.loss_func = torch.nn.CrossEntropyLoss()

        # set some params to non-trainable depending on mode
        apply_train_mode(mode, self.model)

        # for metrics
        self.neutral_prompts = ["Impression: ", "FINDINGS:"]

    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx):
        imgs, labels, tokenized_texts = batch
        image_features = self.encode_image(imgs)
        return image_features
    
    def encode_image(self, imgs):
        return F.normalize(self.model.encode_image(imgs), dim=-1)
    
    def training_step(self, batch, batch_idx):
        imgs, labels, tokenized_texts = batch
        
        out = self.calc_loss(imgs, tokenized_texts, mode="train")
        loss, img_loss, text_loss, img_features, text_features, logits_per_image, logits_per_text = out

        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        lr = self.scheduler.get_last_lr()[0]
        self.log('lr', lr, on_epoch=False, on_step=True, prog_bar=False, logger=True)
        return loss
    
    def on_validation_start(self, *args, **kwargs):
        # calculate label encodings for zero-shot eval
        self.neutral_feats, self.label_feats = calc_zero_shot_embeddings(self.model,
                                                                         device=self.device, 
                                                                         neutral_prompts=self.neutral_prompts,
                                                                         label_names=self.label_names)
        return super().on_validation_start(*args, **kwargs) 

    def validation_step(self, batch, batch_idx):
        imgs, labels, tokenized_texts = batch
        
        out = self.calc_loss(imgs, tokenized_texts, mode="val")
        loss, img_loss, text_loss, img_features, text_features, logits_per_image, logits_per_text = out
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        #self.log('val_img_loss', img_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        #self.log('val_text_loss', text_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        
        neutral_feats = self.neutral_feats.to(img_features.device) 
        all_label_feats = [f.to(img_features.device) for f in self.label_feats]
        # calc zero-shot accuracy
        # first calculate similarities between images and neutral, and images and positive prompts
        # log accuracy first per class then macro-averaged
        for i, prompt in enumerate(self.neutral_prompts):
            neutral_feats_label = neutral_feats[i]
            label_feats = all_label_feats[i]
            cleaned_prompt = prompt.replace(" ", "_")
                        
            zero_shot_class_accuracies = calc_zero_shot_acc(neutral_feats_label, label_feats, img_features, labels)
            self.log('val_zero_shot_acc_macro_' + cleaned_prompt, zero_shot_class_accuracies.mean(), on_epoch=True, prog_bar=True, logger=True)
        # calculate zero-shot accuracy for averaged prompts
        averaged_netral_feats = torch.mean(neutral_feats, dim=0)
        averaged_label_feats = torch.mean(torch.stack(all_label_feats, dim=0), dim=0)
        zero_shot_class_accuracies = calc_zero_shot_acc(averaged_netral_feats, averaged_label_feats, img_features, labels)
        self.log('val_zero_shot_acc_macro_averaged', zero_shot_class_accuracies.mean(), on_epoch=True, prog_bar=True, logger=True)
            
        # calc retrieval at K
        img_retrieval = (torch.argmax(logits_per_image, dim=1) == torch.arange(labels.shape[0], device=labels.device)).float().mean()
        text_retrieval = (torch.argmax(logits_per_text, dim=1) == torch.arange(labels.shape[0], device=labels.device)).float().mean()
        mean_retrieval = (img_retrieval + text_retrieval) / 2
        #self.log('val_retrieval_img', img_retrieval, on_epoch=True, prog_bar=False, logger=True)
        #self.log('val_retrieval_text', text_retrieval, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_retrieval', mean_retrieval, on_epoch=True, prog_bar=True, logger=True)        
        
        return img_features, labels
    
    def validation_epoch_end(self, val_step_output_list):
        return super().validation_epoch_end(val_step_output_list)
    
    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
    
        
    def on_train_epoch_end(self) -> None:
        if self.text2img_num_feats > 0:
            # generate image from text here for some labels - need to enable gradient for it
            #fracs = [0.25, 0.5, 0.75, 1.0]
            fracs = [0.33, 0.66, 1.0]
            eval_epoch_counts = [int(self.max_epochs * frac) - 1 for frac in fracs]
            
            self.epoch_count = self.trainer.current_epoch
            #print("Eval frame:", eval_epoch_counts)
            #print("Epoch count: ", self.epoch_count)
            if self.epoch_count in eval_epoch_counts:
                self.model.eval()
                # generate images
                with torch.set_grad_enabled(True):
                    self.gen_img_from_label()
                # reset
                self.model.train()
        
        return super().on_train_epoch_end()
    
    def on_fit_start(self) -> None:
        if self.text2img_num_feats > 0:
            # generate image from text here for some labels - need to enable gradient for it
            self.model.eval()
            # generate images
            with torch.set_grad_enabled(True):
                self.gen_img_from_label()
            # reset
            self.model.train()
        return super().on_fit_start()
    

    
    def calc_loss(self, imgs, tokenized_texts, mode='train'):
        #if mode == 'train' and self.manifold_mixup:
        #    mixup_idcs = torch.randperm(imgs.shape[0]).to(imgs.device)
        #    mixup_alpha = torch.rand(imgs.shape[0]).to(imgs.device)
        #    mixup_kwargs = {'mixup_idcs': mixup_idcs, 'mixup_alpha': mixup_alpha}
        #    image_features = F.normalize(self.model.encode_image(imgs, **mixup_kwargs), dim=-1)
        #    text_features = F.normalize(self.model.encode_text(tokenized_texts, **mixup_kwargs), dim=-1)
        
        image_features = F.normalize(self.model.encode_image(imgs), dim=-1)
        text_features = F.normalize(self.model.encode_text(tokenized_texts), dim=-1)
        
        if mode == "train":
            # gather representations in case of distributed training
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                image_features = SyncFunction.apply(image_features)
                text_features = SyncFunction.apply(text_features)

        # define labels
        bs = image_features.shape[0]
        ground_truth = torch.arange(bs, dtype=torch.long, device=imgs.device)
            
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
            if self.mixup_alpha > 0 and self.mixup_labels:
                # overwrite labels with mixed labels
                ground_truth = torch.zeros(bs, bs, dtype=torch.float, device=imgs.device)
                ground_truth[torch.arange(bs), mixup_indices] = mixup_strength
                ground_truth[torch.arange(bs), torch.arange(bs)] = 1 - mixup_strength
            
        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        # shape = [global_batch_size, global_batch_size]
        
        img_loss = self.loss_func(logits_per_image, ground_truth)
        text_loss = self.loss_func(logits_per_text, ground_truth)
        loss = (img_loss + text_loss) / 2
        
        
        if self.cyclic_lambda is not None:
            logit_scale = self.model.logit_scale.exp()
            batch_size = imgs.shape[0]
            logits_image_per_image = logit_scale * image_features @ image_features.t()
            logits_text_per_text = logit_scale * text_features @ text_features.t()
            scale = batch_size / logit_scale ** 2
            inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() * scale
            crossmodal_cyclic_loss = (logits_per_image - logits_per_text).square().mean() * scale
            cyclic_loss = self.cyclic_lambda * inmodal_cyclic_loss + self.cyclic_lambda * crossmodal_cyclic_loss
            loss += cyclic_loss
        
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
                      "stack_size": 5,
                     }
        elif net == "vqgan":
            kwargs = {"model_type": "vqgan",
                      "lr": 0.1,
                      "stack_size": 1,
                     }
            
        # disable grads and save state
        grad_state = {n: p.requires_grad for n, p in self.model.named_parameters()}   
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        
        imagine = Imagine(save_gif=False, save_video=False, save_progress=False, verbose=False,
                                  sideX = size, 
                                  sideY = size,
                                  batch_size=self.text2img_batch_size,
                                  clip_models=[self.model],
                                  #clip_names=('ViT-B/16',),
                                  grayscale=True,
                                  tv_loss_scale=0.0,
                                  aesthetic_weight=0.0,
                                  seed=42,
                                  use_russell_transform=True,
                                  use_mixed_precision=True,
                                  optimizer="AdamW", #"MADGRAD",
                                  **kwargs
                                 )
        
        # create imgs
        used_names = self.label_names[:self.text2img_num_feats]
        imgs = []
        pbar = tqdm(used_names)
        for label in pbar:
            pbar.set_description(f"Generating {label}")
            imagine.reset()
            neutral_prompt = "A frontal chest x-ray "
            label_prompt = neutral_prompt + " of " + label
            imagine.set_clip_encoding(text=label_prompt)
            for _ in range(self.text2img_num_steps):
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
        for n, p in self.model.named_parameters():
            p.requires_grad_(grad_state[n])
        
        #apply_train_mode(self.mode, self.model)

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
            model = LogisticRegression(solver="saga", max_iter=200)
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