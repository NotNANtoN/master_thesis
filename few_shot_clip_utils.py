import torch
# model, transform

# 1. only fine-tune mlp on top of frozen-features
# 2. (1.) + fine-tune layer-norms

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() 


class CLIPClassifier(torch.nn.Module):
    def __init__(self, model, mode, num_labels):
        super().__init__()
        self.mode = mode
        
        # create layers
        self.model = model
        out_feat_size = model.text_projection.shape[1]
        self.out = torch.nn.Linear(out_feat_size, num_labels)
        
        if mode == "freeze":
            # delete layers as we now just feed in the precomputed features
            #self.model = model.clone()
            #self.model.encode_image = torch.nn.Identity()
            #self.model.encode_text = torch.nn.Identity()
            for p in self.model.parameters():
                p.requires_grad = False
        elif mode == "train_norm":
            for n, p in self.model.named_parameters():
                p.requires_grad = ".ln_" in n or ".bn_" in n

    def forward(self, x):
        x = self.encode_image(x)
        x = self.out(x)
        return x
    
    def encode_image(self, x):
        if mode == "freeze":
            return x
        else:
            return self.model.encode_image(x)
    
    def encode_text(self, x):
        if mode == "freeze":
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


    
@torch.inference_mode()
def val_step(model, val_dl):
    episode_val_losses = []
    all_preds = []
    all_y = []
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        # calc loss
        preds = clip_classifier(x)
        loss = loss_func(preds, y)
        # log
        episode_val_losses.append(loss)
        all_preds.append(preds)
        all_y.append(y)
    all_preds = torch.sigmoid(torch.cat(all_preds, dim=0)).cpu().numpy()
    all_y = torch.cat(all_y, dim=0).long().cpu().numpy()
    val_losses.append(torch.stack(episode_val_losses).mean().item())
    
    all_preds_binary = binarize_preds(all_y, all_preds)
    accuracies = [(all_y[:, idx] == all_preds_binary[:, idx]).mean() 
                       for idx in range(all_y.shape[1])]
    accs = [sklearn.metrics.accuracy_score(all_y[:, label_idx], all_preds_binary[:, label_idx])
       for label_idx in range(all_y.shape[1])]
    aps = [sklearn.metrics.average_precision_score(all_y[:, label_idx], all_preds[:, label_idx], average='macro', pos_label=1)
       for label_idx in range(all_y.shape[1])]
    val_accs.append(np.mean(accuracies))
    val_aps.append(np.mean(aps))
    
