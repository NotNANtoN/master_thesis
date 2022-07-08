import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers.adapters.modeling import Adapter


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

############### classes to modify to add adapters to CLIP
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    
class Transformer(nn.Module):
    # clip has one transformer directly in it for text and another is visual.transformer
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
##################

class ResidualAttentionBlockAdapters(ResidualAttentionBlock):
    def __init__(self, d_model: int, n_head: int, down_sample: int, attn_mask: torch.Tensor = None,
                 adapter_flow="hard"):
        super().__init__(d_model, n_head, attn_mask=attn_mask)
        self.adapter_flow = adapter_flow
        self.d_model = d_model
        # adapter
        #self.adapter = torch.nn.Sequential(torch.nn.Linear(d_model, d_adapter),
        #                                   torch.nn.Linear(d_adapter, d_model))
        self.adapter_kwargs = {"input_size": d_model, "down_sample": down_sample,
                               "non_linearity": "relu", "init_bert_weights": True, "add_layer_norm_before": True,
                               "add_layer_norm_after": False, "residual_before_ln": True}
                               
        self.adapter = Adapter(**self.adapter_kwargs)                      
        #self.adapter =  Adapter(d_model, down_sample=down_sample, non_linearity='relu',
        #                        init_bert_weights=True, add_layer_norm_before=True, 
        #                        add_layer_norm_after=False, residual_before_ln=True)
        
        self.extra_adapters = nn.ModuleList()
    def forward(self, x: torch.Tensor):
        # regular forward:
        #x = x + self.attention(self.ln_1(x))
        #x = x + self.mlp(self.ln_2(x))
        # or
        # att_out = x + self.attention(self.ln_1(x))
        # mlp_out = self.mlp(self.ln_2(att_out))
        # x = att_out + mlp_out
        
        if self.adapter_flow == "hard":
            # derived from flow of Pfeiffer et al.
            att_out = x + self.attention(self.ln_1(x))
            mlp_out = self.mlp(self.ln_2(att_out))
            adapter_out = self.adapter(att_out + mlp_out, mlp_out)[0]
            # equivalent to: adapter_out = self.adapter(att_out + mlp_out)[0] + mlp_out
            x = att_out + adapter_out
        
        elif self.adapter_flow == "easy":
            # easier variation:
            # att_out = x + self.attention(self.ln_1(x))
            # x = att_out + self.mlp(self.ln_2(att_out))
            # x = self.adapter(x, x)[0]
            #  or
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            x = self.adapter(x, x)[0]
            for extra_adapter in self.extra_adapters:
                x = extra_adapter(x, x)[0]
        elif self.adapter_flow == "correct":
            x = x + self.attention(self.ln_1(x))
            mlp_out = self.mlp(self.ln_2(x))
            x = x + self.adapter(mlp_out, mlp_out)[0]
            
            for extra_adapter in self.extra_adapters:
                x = extra_adapter(x, x)[0]
        return x
    
    def add_adapter(self):
        # first set all weights so far to not trainable except layernorms
        for n, p in self.named_parameters():
            p.requires_grad = "ln_" in n
        # add adapter
        adapter = Adapter(**self.adapter_kwargs) 
        self.extra_adapters.append(adapter)


class TransformerWithAdapters(nn.Module):
    # clip has one transformer directly in it for text and another is visual.transformer
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                down_sample: int = 16, adapter_flow="hard"):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockAdapters(width, heads, 
                                                                        down_sample, attn_mask,
                                                                        adapter_flow=adapter_flow)
                                         for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
    def add_adapter(self):
        for resblock in self.resblocks:
            resblock.add_adapter()
    
    
def add_adapters(model, down_sample_size=16, adapter_flow="hard"):
    adapter_model = copy.deepcopy(model)
    weights = model.state_dict()
    # replace transformers by adapters with transformers
    adapter_model.transformer = TransformerWithAdapters(model.transformer.width, 
                                                model.transformer.layers,
                                                model.transformer.resblocks[0].attn.num_heads,
                                                attn_mask=model.build_attention_mask(),
                                                down_sample=down_sample_size,
                                                adapter_flow=adapter_flow,
                                               )
    adapter_model.visual.transformer = TransformerWithAdapters(model.visual.transformer.width,
                                                       model.visual.transformer.layers,
                                                       model.visual.transformer.resblocks[0].attn.num_heads,
                                                       down_sample=down_sample_size,
                                                       adapter_flow=adapter_flow,
                                                      )
    # set the weights of the transformers (except the adapter layers)
    adapter_model.load_state_dict(weights, strict=False)
    # set only adapter and ln weights to trainable
    for n, p in adapter_model.transformer.named_parameters():
        p.requires_grad = "adapter" in n or "ln" in n
    for n, p in adapter_model.visual.transformer.named_parameters():
        p.requires_grad = "adapter" in n or "ln" in n
    return adapter_model


def add_adapters_visual(model, down_sample_size=16, adapter_flow="hard"):
    adapter_model = copy.deepcopy(model)
    weights = model.state_dict()
    # replace transformers by adapters with transformers
    adapter_model.transformer = TransformerWithAdapters(adapter_model.transformer.width,
                                    adapter_model.transformer.layers,
                                    adapter_model.transformer.resblocks[0].attn.num_heads,
                                    down_sample=down_sample_size,
                                    adapter_flow=adapter_flow,
                                    )
    # set the weights of the transformers (except the adapter layers)
    adapter_model.load_state_dict(weights, strict=False)
    # set only adapter and ln weights to trainable
    for n, p in adapter_model.transformer.named_parameters():
        p.requires_grad = "adapter" in n or "ln" in n
    return adapter_model
    
#down_sample = 4
#adapter = Adapter(input_size, down_sample=down_sample, non_linearity='relu', init_bert_weights=True, add_layer_norm_before=True, add_layer_norm_after=False, residual_before_ln=True)

#transformers.PfeifferConfig(original_ln_before: bool = True, original_ln_after: bool = True, residual_before_ln: bool = True, adapter_residual_before_ln: bool = False, ln_before: bool = False, ln_after: bool = False, mh_adapter: bool = False, output_adapter: bool = True, non_linearity: str = 'relu', reduction_factor: Union[int, collections.abc.Mapping] = 16, inv_adapter: Optional[str] = None, inv_adapter_reduction_factor: Optional[int] = None, cross_adapter: bool = False, leave_out: List[int] = <factory>)