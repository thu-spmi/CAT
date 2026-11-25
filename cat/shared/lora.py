import math
import re
import torch
from torch import nn
import torch.nn.functional as F

# from peft import PeftModel
# from peft.tuners.lora.model import LoraLayer

class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias=None):
        super(LinearLayer_LoRA, self).__init__()

       
        self.weight = weight

        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )
        self.lora = LoRA(self.weight,lora_dim,lora_scaling,lora_dropout)
        
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora.lora_scaling * torch.matmul(
                self.lora.lora_left_weight.t(), self.lora.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora.lora_scaling * torch.matmul(
                self.lora.lora_left_weight.t(), self.lora.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            # print(f'input_dtype: {input.dtype}')
            # print(f'weight_dtype: {self.weight.dtype}')
            return F.linear(input, self.weight, self.bias)
        else:
            freeze_output = F.linear(
            input, self.weight,
            self.bias)
            return freeze_output + self.lora(input)

class LoRA(nn.Module):
    def __init__(self, weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,):
        super(LoRA, self).__init__()
        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
    def eval(self):
        self.lora_dropout.eval()
    def train(self, mode=True):
        self.lora_dropout.train(mode)
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)
    def forward(self, input):
        return (self.lora_dropout(input) @ self.lora_right_weight
                            @ self.lora_left_weight) * self.lora_scaling





# Helper functions
def recursive_getattr(obj, attr):
    """Recursively get attributes from an object."""
    attrs = attr.split(".")
    for i in attrs:
        obj = getattr(obj, i)
    return obj


def recursive_setattr(obj, attr, value):
    """Recursively set attributes to an object."""
    attrs = attr.split(".")
    for i in attrs[:-1]:
        obj = getattr(obj, i)
    setattr(obj, attrs[-1], value)



# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model,del_lora_model=False,use_peft=False):
    if not use_peft:
        replace_name = []
        for name, module in model.named_modules():
            if isinstance(module, LinearLayer_LoRA):
                replace_name.append(name)
        for name in replace_name:
            module = recursive_getattr(model, name)
            # zero_stage_3 = hasattr(module.weight, 'ds_id')
            # with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
            #         module.weight, module.bias, module.lora_left_weight,
            #         module.lora_right_weight
            # ]),
            #                                        modifier_rank=0,
            #                                        enabled=zero_stage_3):
            module.fuse_lora_weight()
            if del_lora_model:
                del module.lora
    else:
        for name, child in module.named_children():
            # 如果是 PEFT 包装的模型
            if isinstance(child, PeftModel):
                # 融合 LoRA 到 base model 并卸载 LoRA
                fused_model = child.merge_and_unload()
                setattr(module, name, fused_model)  # 替换原来的子模块
            else:
                # 递归处理子模块
                convert_peft_lora_to_base_model(child, del_lora_model=del_lora_model)
    return model

# convert the linear layer to LoRA, change name matching strategy (zhaoxy)
def convert_linear_layer_to_loraX(model,
                                 part_module_name,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_dropout=0,
                                 task_type="CAUSAL_LM",
                                 use_peft=False):
    """
    Convert linear layers in a model to use LoRA.

    This function supports two types of LoRA injection:
      1. LinearLayer-based LoRA (manual implementation)
      2. PEFT-based LoRA (using HuggingFace PEFT library)

    Parameters
    ----------
    model : torch.nn.Module
        The model to inject LoRA into. Can be a standard HF transformers model
        or a custom model with linear layers.
    part_module_name : str or list[str]
        List of module names to apply LoRA to (e.g., ["q_proj", "v_proj"]).
    lora_dim : int, default=0
        LoRA rank (low-rank dimension).
    lora_scaling : float, default=1
        LoRA scaling factor (alpha).
    lora_droppout : float, default=0
        Dropout probability for LoRA layers.
    task_type : str, default="CAUSAL_LM"
        Task type for PEFT LoRA. Options include "CAUSAL_LM" or "SEQ_2_SEQ_LM".
    use_peft : bool, default=False
        Whether to use PEFT for LoRA injection.

    Notes for PEFT-based LoRA
    -------------------------
    1. PEFT wraps the original model. It is recommended to inject LoRA only 
       on standard HuggingFace transformer models.
    2. If you want to enable gradient checkpointing, it **must** be enabled 
       before wrapping the model with PEFT.
       Reference from PEFT documentation:
           "When using LoRA with gradient checkpointing, 
            you must enable gradient checkpointing before wrapping the model with PEFT."
    """
    replace_name = []
    if isinstance(part_module_name, str):
        assert '^' in part_module_name
        part_module_name = part_module_name.split('^')

    # if not use_peft:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _check_target_module_exists(part_module_name, name):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_scaling, lora_dropout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    model = only_optimize_lora_parameters(model)
    model = make_model_gradient_checkpointing_compatible(model)
    # else:
    #     from peft import LoraConfig, get_peft_model
    #     lora_config = LoraConfig(
    #         r=lora_dim,                # LoRA rank
    #         lora_alpha=lora_scaling,       # LoRA 缩放系数
    #         target_modules=part_module_name,  
    #         lora_dropout=lora_dropout,
    #         task_type="CAUSAL_LM"
    #     )
    #     model = get_peft_model(model, lora_config)
    #     # if torch.distributed.get_rank() == 0:
    #     #     model.print_trainable_parameters()
    #     #     print(model)
    return model




def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model


def _check_target_module_exists(part_module_name, key):
    if isinstance(part_module_name, str):
        target_module_found = re.fullmatch(part_module_name, key)
    else:
        target_module_found = any(
            re.match(f".*\.{target_key}$", key) for target_key in part_module_name
        ) or any(target_key == key for target_key in part_module_name)    
    return target_module_found


def convert_lora_freeze_weight_to_bf16(module):
    """
    Recursively traverse the given module and convert LoRA weight tensors to bfloat16 (bf16).

    Supports:
    1. Custom LinearLayer_LoRA
    2. PEFT-based LoraLayer (from HuggingFace PEFT)

    Parameters
    ----------
    module : torch.nn.Module
        The model or submodule to traverse.
    """
    for child_name, child in module.named_children():
        if isinstance(child, LinearLayer_LoRA):
            child.weight.data = child.weight.data.to(torch.bfloat16)
        # elif isinstance(child, LoraLayer):
            # if hasattr(child, "lora_A") and isinstance(child.lora_A, torch.nn.Parameter):
                # child.lora_A.data = child.lora_A.data.to(torch.bfloat16)
            # if hasattr(child, "lora_B") and isinstance(child.lora_B, torch.nn.Parameter):
                # child.lora_B.data = child.lora_B.data.to(torch.bfloat16)
        else:
            # 如果是 ModuleDict，递归遍历它的子模块
            convert_lora_freeze_weight_to_bf16(child)