import torch
from PIL import Image
import clip
import pytorch_lightning

from clip_utils import apply_train_mode


class LitCLCLIP(pytorch_lightning.LightningModule):
    def __init__(self, model, mode, max_epochs, learning_rate, steps_per_epoch, 
                 weight_decay=0.2,
                 gen_freq=5,  # after how many val epochs there should be an image generation
                ):
        super().__init__()
        # args
        self.mode = mode
        self.save_hyperparameters()
        self.max_epochs = max_epochs
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps_per_epoch = steps_per_epoch
        self.gen_freq = gen_freq

        #
        apply_train_mode(mode, self.model)
        
        self.gen_count = 0
        # loss func
        self.loss_func = torch.nn.CrossEntropyLoss()
        # metrics
                
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        imgs, labels, tokenized_texts = batch
        loss, img_loss, text_loss, img_features, text_features, logits_per_image, logits_per_text = self.calc_loss(imgs, tokenized_texts)
        
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        lr = self.scheduler.get_last_lr()[0]
        self.log('lr', lr, on_epoch=False, on_step=True, prog_bar=False, logger=True)
        return loss
    
    def on_validation_start(self, *args, **kwargs):
        super().on_validation_start(*args, **kwargs)
        # calculate label encodings for zero-shot eval
        neutral_prompt = "An X-ray scan of the chest of a person"
        label_prompts = [neutral_prompt + " with a " + label for label in self.label_names]
        tokenized_neutral = clip.tokenize([neutral_prompt]).to(self.device)
        tokenized_labels = clip.tokenize(label_prompts).to(self.device)
        label_feats = self.model.encode_text(tokenized_labels)
        neutral_feats = self.model.encode_text(tokenized_neutral)
        self.label_feats = label_feats / label_feats.norm(dim=-1, keepdim=True)
        self.neutral_feats = neutral_feats / neutral_feats.norm(dim=-1, keepdim=True)
    
    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        # generate image from text here for some labels - need to enable grade for it
        if self.gen_count % self.gen_freq == 0:
            with torch.set_grad_enabled(True):
                self.gen_img_from_label()
        # TODO: add options of how many steps and how many labels
        self.gen_count += 1
            
    
    def validation_step(self, batch, batch_idx):
        imgs, labels, tokenized_texts = batch
        loss, img_loss, text_loss, img_features, text_features, logits_per_image, logits_per_text = self.calc_loss(imgs, tokenized_texts)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        #self.log('val_img_loss', img_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        #self.log('val_text_loss', text_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        
        # calc zero-shot accuracy
        # first calculate similarities between images and neutral, and images and positive prompts
        neutral_sim = torch.cosine_similarity(self.neutral_feats, img_features)
        label_sims = torch.stack([torch.cosine_similarity(label_feat.unsqueeze(0), img_features) for label_feat in self.label_feats])
        self.log('val_neutral_mean_sim', neutral_sim.mean(), on_epoch=True, prog_bar=False, logger=True)
        self.log('val_label_mean_sim', label_sims.mean(), on_epoch=True, prog_bar=False, logger=True)
        binary_preds = torch.stack([label_sim > neutral_sim for label_sim in label_sims], dim=1).int()
        # log accuracy first per class then macro-averaged
        zero_shot_class_accuracies = (labels.int() == binary_preds).float().mean(dim=0)
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
        
        return {"loss": loss}
    
    def calc_loss(self, imgs, tokenized_texts):
        bs = imgs.shape[0]
        logits_per_image, logits_per_text, img_features, text_features = self.clip_forward(imgs, tokenized_texts)
        ground_truth = torch.arange(bs, dtype=torch.long, device=imgs.device)
        
        img_loss = self.loss_func(logits_per_image, ground_truth)
        text_loss = self.loss_func(logits_per_text, ground_truth)
        loss = (img_loss + text_loss) / 2
        return loss, img_loss, text_loss, img_features, text_features, logits_per_image, logits_per_text
    
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

    def clip_forward(self, image, text):
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text, image_features, text_features
    
    def gen_img_from_label(self):
        # init text2img
        import sys
        sys.path.append("../CLIPGuidance/")
        from style_clip import Imagine
        
        net = "image"
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
        for p in self.model.parameters():
            p.requires_grad_(False)
        imagine = Imagine(save_gif=False, save_video=False, save_progress=False, verbose=False,
                                  sideX = 256, 
                                  sideY = 256,
                                  batch_size=8,
                                  clip_models=[self.model],
                                  #clip_names=('ViT-B/16',),
                                  grayscale=True,
                                  tv_loss_scale=1.0,
                                  seed=42,
                                  use_russell_transform=True,
                                  use_mixed_precision=True,
                                  optimizer="AdamW", #"MADGRAD",
                                  **kwargs
                                 )
        
        # create imgs
        used_names = self.label_names[:4]
        neutral_prompt = "An X-ray scan of the chest of a person"
        imgs = []
        for label in used_names:
            imagine.reset()
            label_prompt = neutral_prompt + " with a " + label
            imagine.set_clip_encoding(text=label_prompt) #"FINDING: The patient has a " + label)
            for i in range(200):
                img, loss = imagine.train_step(0, 0)
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
