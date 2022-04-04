import torch
import numpy as np
from PIL import Image
import pytorch_lightning
import torchmetrics
import pl_bolts 
import torchvision.transforms as TF
from sklearn.metrics import f1_score, accuracy_score
import sklearn

from clip_utils import get_clip_img_caption_features, apply_train_mode


def convert_models_to_fp32(model):
    #https://github.com/openai/CLIP/issues/57
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() 

            
class LitCLIP(pytorch_lightning.LightningModule):
    def __init__(self, model, max_epochs, learning_rate, steps_per_epoch, 
                 weight_decay=0.2, use_pos_weight=True, pos_fraction=0.5):
        super().__init__()
        # args
        self.save_hyperparameters()
        self.max_epochs = max_epochs
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps_per_epoch = steps_per_epoch
        self.eps = 1e-6
        self.betas = (0.9, 0.98)
        # loss func
        neg_fraction = 1 - pos_fraction
        pos_weight = neg_fraction / pos_fraction if use_pos_weight else None
        if use_pos_weight:
            print("Pos weight: ", pos_weight)
        self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # metrics
        self.label_names = model.label_names
        #self.auroc = torchmetrics.AUROC(num_classes=model.num_labels, average=None)
        #self.ap = torchmetrics.AveragePrecision(num_classes=model.num_labels, average="macro")
        #self.f1 = torchmetrics.F1Score(num_classes=model.num_labels)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self.calc_loss(x, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        lr = self.scheduler.get_last_lr()[0]
        self.log('lr', lr, on_epoch=False, on_step=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self.calc_loss(x, y)
        preds = torch.sigmoid(preds)
        y = y.int()
        
        #self.auroc.update(preds, y)
        #self.ap.update(preds, y)
        #self.log('val_ap', self.ap, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return preds, y
    
    def validation_epoch_end(self, val_step_output_list):
        # calculate f1_score, accuracy etc
        preds = torch.cat([step_out[0] for step_out in val_step_output_list]).cpu().numpy()#.reshape(-1)
        targets = torch.cat([step_out[1] for step_out in val_step_output_list]).cpu().numpy()#.reshape(-1)
        
        # average precision
        ap = sklearn.metrics.average_precision_score(targets, preds, average="macro", pos_label=1)
        self.log('val_ap', ap, on_epoch=True, prog_bar=True)
        # auc - per class and macro-averaged
        #print("Shapes: ", targets.shape, preds.shape)
        class_aucs = sklearn.metrics.roc_auc_score(targets, preds, average=None)
        #print("AUCS: ", class_aucs)
        auc_dict = {"val_auc/" + self.label_names[i]: class_aucs[i]
                    for i in range(len(class_aucs))}
        self.log_dict(auc_dict, on_epoch=True)
        self.log('val_auroc_macro', class_aucs.mean(), on_epoch=True, prog_bar=True)
        self.log('val_auroc_micro', sklearn.metrics.roc_auc_score(targets, preds, average="micro"),
                  on_epoch=True, prog_bar=True)
        
        # metrics based on binary predictions
        binary_preds = (preds > 0.5).astype(int)
        self.log("val_acc_micro", accuracy_score(targets.reshape(-1), binary_preds.reshape(-1)), on_epoch=True)
        macro_acc = (targets == binary_preds).astype(float).mean(axis=0).mean()
        self.log("val_acc_macro", macro_acc, on_epoch=True)
        self.log("val_f1_macro", f1_score(targets, binary_preds, average="macro"), on_epoch=True, logger=True)
        self.log("val_f1_micro", f1_score(targets, binary_preds, average="micro"), on_epoch=True, logger=True)
        
        # log diagnostics of output distribution
        preds_for_pos = preds[targets == 1]
        preds_for_neg = preds[targets == 0]
        pos_mean = preds_for_pos.mean()
        neg_mean = preds_for_neg.mean()
        self.log("debug/pos_preds_mean", pos_mean, on_epoch=True)
        self.log("debug/pos_preds_std", preds_for_pos.std(), on_epoch=True)
        self.log("debug/neg_preds_mean", neg_mean, on_epoch=True)
        self.log("debug/neg_preds_std", preds_for_neg.std(), on_epoch=True)
        self.log("debug/preds_mean", preds.mean(), on_epoch=True)
        self.log("debug/preds_std", preds.std(), on_epoch=True)
        self.log("debug/preds_mean_diff", pos_mean - neg_mean, on_epoch=True)
    
    def calc_loss(self, x, y):
        preds = self(x)
        loss = self.loss_func(preds, y)
        return loss, preds
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                              lr=self.learning_rate,#5e-5,
                              betas=self.betas,
                              eps=self.eps,
                              weight_decay=self.weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.learning_rate, steps_per_epoch=self.steps_per_epoch,
                                                             epochs=self.max_epochs, pct_start=0.05)
        scheduler = {"scheduler": self.scheduler, 
                     "interval": "step" }  # necessary to make PL call the scheduler.step after every batch
        return [optimizer], [scheduler]
    

class CLIPClassifier(torch.nn.Module):
    def __init__(self, model, mode, num_labels):
        super().__init__()
        self.mode = mode
        self.num_labels = num_labels
        
        # create layers
        self.model = model
        out_feat_size = model.text_projection.shape[1]
        self.out = torch.nn.Linear(out_feat_size, num_labels)
        # delete text part
        if hasattr(model, "transformer"):
            del model.transformer
            
        apply_train_mode(mode, self.model)

    def forward(self, x):
        x = self.encode_image(x)
        x = self.out(x)
        return x
    
    def encode_image(self, x):
        if self.mode == "freeze":
            return x
        else:
            return self.model.encode_image(x)
    
    def encode_text(self, x):
        if self.mode == "freeze":
            return x
        else:
            return self.model.encode_text(x)

        

def binarize_preds(all_y, all_preds):
    best_thresholds = []
    for idx in range(all_y.shape[1]):
        thresholds = np.arange(0, 1, 0.05)
        accs = [sklearn.metrics.accuracy_score(all_y[:, idx], all_preds[:, idx] > thresh)
                for thresh in thresholds] 
        #fprs, tprs, thresholds = sklearn.metrics.roc_curve(all_y[:, idx], all_preds[:, idx])
        best_thresh_idx = np.argmax(accs)
        best_thresh = thresholds[best_thresh_idx]
        best_thresholds.append(best_thresh)
    all_preds_binary = np.array([all_preds[:, idx] > best_thresholds[idx] for idx in range(all_y.shape[1])]).reshape(*all_y.shape)
    return all_preds_binary


def binarize_preds_roc(all_y, all_preds):
    best_thresholds = []
    for idx in range(all_y.shape[1]):
        fprs, tprs, thresholds = sklearn.metrics.roc_curve(all_y[:, idx], all_preds[:, idx])
        best_thresh_idx = np.argmax(tprs - fprs)
        best_thresh = thresholds[best_thresh_idx]
        best_thresholds.append(best_thresh)
    all_preds_binary = np.array([all_preds[:, idx] > best_thresholds[idx] for idx in range(all_y.shape[1])]).reshape(*all_y.shape)
    return all_preds_binary
