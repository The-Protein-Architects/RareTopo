import logging

import torch
import torch.nn as nn

logger = logging.getLogger("TopoDiff.model.lora")

class LoRALinear(nn.Module):
    """LoRA adapter for Linear layers.

    The wrapped base layer is kept frozen and the low-rank update is trained.
    """

    def __init__(self, base_layer, r=8, alpha=16, dropout=0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

def _replace_child(parent, child_name, new_child):
    if isinstance(parent, nn.ModuleDict):
        parent[child_name] = new_child
    elif isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = new_child
    else:
        setattr(parent, child_name, new_child)

def apply_lora_to_linear(model, target_keywords=None, r=8, alpha=16, dropout=0.05):
    """Replace selected Linear layers with LoRA-wrapped Linear layers.

    Args:
        model: torch module to modify in place.
        target_keywords: layer names must contain at least one of these strings.
        r: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout during training.

    Returns:
        List of replaced module names.
    """
    if target_keywords is None:
        target_keywords = ["backbone"]

    if isinstance(target_keywords, str):
        target_keywords = [target_keywords]

    replaced = []

    for module_name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue

        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if not any(keyword in full_name for keyword in target_keywords):
                continue
            # MultiheadAttention.forward reads out_proj.weight/out_proj.bias
            # directly, so replacing it with a wrapper module breaks PyTorch's
            # implementation. Keep it frozen as part of the base model.
            if isinstance(module, nn.MultiheadAttention) and child_name == "out_proj":
                continue
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                _replace_child(
                    module,
                    child_name,
                    LoRALinear(child, r=r, alpha=alpha, dropout=dropout),
                )
                replaced.append(full_name)

    return replaced

def mark_only_lora_as_trainable(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name

def count_trainable_parameters(model):
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total

def lora_state_dict(model):
    return {
        key: value
        for key, value in model.state_dict().items()
        if "lora_A" in key or "lora_B" in key
    }
